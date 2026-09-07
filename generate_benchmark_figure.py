#!/usr/bin/env python3
"""
generate_benchmark_figure.py
Generates a publication-grade, high-legibility multi-panel benchmark comparison
for the TCC monograph, saving to experiments/05_09_2026_run2/benchmark_comparison.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_figure():
    exp_dir = Path("experiments/05_09_2026_run2")
    csv_path = exp_dir / "master_leaderboard.csv"
    out_path = exp_dir / "benchmark_comparison.png"

    df = pd.read_csv(csv_path)

    # Filter main scenario runs and remove duplicates/backups
    main_df = df[(df['scenario'] == 'main') & (df['duplicate_of'].isna() | (df['duplicate_of'] == ''))].copy()

    # Define architectures and metadata
    archs = ['v1', 'v2', 'v3', 'v4', 'v5']
    arch_labels = ['V1\n(3.2M)', 'V2\n(2.2M)', 'V3\n(4.2M)', 'V4\n(8.4M)', 'V5\n(294.1M)']
    params_m = [3.2, 2.2, 4.2, 8.4, 294.1]

    # Compute mean and std across canonical seeds (42, 43, 44) for each architecture
    stats = {}
    for m in archs:
        m_runs = main_df[(main_df['model'] == m) & (main_df['seed'].isin([42, 43, 44]))]
        stats[m] = {
            'ham_mel_auc_mean': m_runs['ham_mel_auc_roc'].mean(),
            'ham_mel_auc_std': m_runs['ham_mel_auc_roc'].std(),
            'pad_mel_auc_mean': m_runs['mel_auc_roc'].mean(),
            'pad_mel_auc_std': m_runs['mel_auc_roc'].std(),
            'pad_macro_auc_mean': m_runs['pad_macro_auc_roc'].mean(),
            'pad_macro_auc_std': m_runs['pad_macro_auc_roc'].std(),
            'pad_sens_mean': m_runs['pad_mal_sens_gated'].mean(),
            'pad_sens_std': m_runs['pad_mal_sens_gated'].std(),
            'pad_acc_mean': m_runs['pad_restricted_5class_acc'].mean(),
            'pad_acc_std': m_runs['pad_restricted_5class_acc'].std(),
            'ham_ece_mean': m_runs['ham_ece'].mean(),
            'ham_ece_std': m_runs['ham_ece'].std(),
            'pad_ece_mean': m_runs['pad_ece'].mean(),
            'pad_ece_std': m_runs['pad_ece'].std(),
            'seeds': m_runs[['seed', 'ham_mel_auc_roc', 'mel_auc_roc', 'pad_macro_auc_roc', 'pad_mal_sens_gated', 'pad_ece']].to_dict('records')
        }

    # Style configuration
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    fig.patch.set_facecolor('#ffffff')

    x = np.arange(len(archs))
    width_2 = 0.36
    width_3 = 0.26

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 1 (Top-Left): Domain Shift: HAM10000 vs. PAD-UFES-20 Melanoma AUC
    # ─────────────────────────────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor('#fafafa')

    ham_auc_means = [stats[m]['ham_mel_auc_mean'] for m in archs]
    ham_auc_stds = [stats[m]['ham_mel_auc_std'] for m in archs]
    pad_auc_means = [stats[m]['pad_mel_auc_mean'] for m in archs]
    pad_auc_stds = [stats[m]['pad_mel_auc_std'] for m in archs]

    bars1_ham = ax1.bar(x - width_2/2, ham_auc_means, width_2, yerr=ham_auc_stds,
                        capsize=4, label='In-Domain: HAM10000 (Dermoscopia)',
                        color='#1f77b4', edgecolor='#0f4c81', linewidth=1.2, alpha=0.92)
    bars1_pad = ax1.bar(x + width_2/2, pad_auc_means, width_2, yerr=pad_auc_stds,
                        capsize=4, label='Out-of-Domain: PAD-UFES-20 (Smartphone)',
                        color='#e6550d', edgecolor='#a63603', linewidth=1.2, alpha=0.92)

    # Scatter points for individual seeds
    for i, m in enumerate(archs):
        for rec in stats[m]['seeds']:
            ax1.scatter(i - width_2/2 + (np.random.rand()-0.5)*0.06, rec['ham_mel_auc_roc'],
                        color='#08306b', s=28, zorder=5, alpha=0.8)
            ax1.scatter(i + width_2/2 + (np.random.rand()-0.5)*0.06, rec['mel_auc_roc'],
                        color='#7f2704', s=28, zorder=5, alpha=0.8)

    # Annotations
    for bar in bars1_ham:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.015),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#08306b')
    for bar in bars1_pad:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.015),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#a63603')

    ax1.set_title('(A) Degradação de Transferência: AUC-ROC de Melanoma', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('AUC-ROC [0.50 - 1.00]', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, fontsize=11, fontweight='bold')
    ax1.set_ylim(0.50, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 2 (Top-Right): PAD-UFES-20 Clinical & Multi-Class Metrics
    # ─────────────────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor('#fafafa')

    pad_macro_means = [stats[m]['pad_macro_auc_mean'] for m in archs]
    pad_macro_stds = [stats[m]['pad_macro_auc_std'] for m in archs]
    pad_sens_means = [stats[m]['pad_sens_mean'] for m in archs]
    pad_sens_stds = [stats[m]['pad_sens_std'] for m in archs]
    pad_acc_means = [stats[m]['pad_acc_mean'] for m in archs]
    pad_acc_stds = [stats[m]['pad_acc_std'] for m in archs]

    bars2_macro = ax2.bar(x - width_3, pad_macro_means, width_3, yerr=pad_macro_stds,
                          capsize=3, label='PAD Macro AUC (5 classes)',
                          color='#2ca02c', edgecolor='#1a661a', linewidth=1.2, alpha=0.9)
    bars2_sens = ax2.bar(x, pad_sens_means, width_3, yerr=pad_sens_stds,
                         capsize=3, label='PAD Sensibilidade Maligna',
                         color='#756bb1', edgecolor='#54278f', linewidth=1.2, alpha=0.9)
    bars2_acc = ax2.bar(x + width_3, pad_acc_means, width_3, yerr=pad_acc_stds,
                        capsize=3, label='PAD Acurácia (5 classes)',
                        color='#3182bd', edgecolor='#08519c', linewidth=1.2, alpha=0.9)

    for bar in bars2_macro:
        h = bar.get_height()
        ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.015),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a661a')
    for bar in bars2_sens:
        h = bar.get_height()
        ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.015),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#54278f')
    for bar in bars2_acc:
        h = bar.get_height()
        ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.015),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#08519c')

    ax2.set_title('(B) Métricas Clínicas e Multiclasse no PAD-UFES-20', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel('Pontuação da Métrica [0.0 - 1.0]', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, fontsize=11, fontweight='bold')
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 3 (Bottom-Left): Calibration Error (ECE): In-Domain vs. Out-of-Domain
    # ─────────────────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor('#fafafa')

    ham_ece_means = [stats[m]['ham_ece_mean'] for m in archs]
    ham_ece_stds = [stats[m]['ham_ece_std'] for m in archs]
    pad_ece_means = [stats[m]['pad_ece_mean'] for m in archs]
    pad_ece_stds = [stats[m]['pad_ece_std'] for m in archs]

    bars3_ham = ax3.bar(x - width_2/2, ham_ece_means, width_2, yerr=ham_ece_stds,
                        capsize=4, label='In-Domain ECE (HAM10000)',
                        color='#41b6c4', edgecolor='#225ea8', linewidth=1.2, alpha=0.9)
    bars3_pad = ax3.bar(x + width_2/2, pad_ece_means, width_2, yerr=pad_ece_stds,
                        capsize=4, label='Out-of-Domain ECE (PAD-UFES-20)',
                        color='#cb181d', edgecolor='#67000d', linewidth=1.2, alpha=0.9)

    for bar in bars3_ham:
        h = bar.get_height()
        ax3.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.012),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#225ea8')
    for bar in bars3_pad:
        h = bar.get_height()
        ax3.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h + 0.012),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#67000d')

    ax3.set_title('(C) Erro de Calibração Esperado (ECE) — Menor é Melhor', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel('ECE [0.0 - 0.6]', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_labels, fontsize=11, fontweight='bold')
    ax3.set_ylim(0.0, 0.60)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 4 (Bottom-Right): Efficiency Trade-off: Parameters vs. PAD Melanoma AUC
    # ─────────────────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#fafafa')

    colors_scatter = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    arch_names_clean = ['MobileNetV1', 'MobileNetV2', 'MobileNetV3', 'MobileNetV4', 'MobileNetV5']

    for i, m in enumerate(archs):
        p = params_m[i]
        auc_m = stats[m]['pad_mel_auc_mean']
        auc_s = stats[m]['pad_mel_auc_std']

        ax4.errorbar(p, auc_m, yerr=auc_s, fmt='o', color=colors_scatter[i],
                     markersize=12, capsize=6, elinewidth=2, markeredgecolor='black', markeredgewidth=1.2,
                     label=f'{arch_names_clean[i]} ({p}M): {auc_m:.3f}')

        # Text offsets tailored for clear separation
        dx = 1.15
        dy = 0.006 if m in ['v2', 'v5'] else -0.014
        if m == 'v1':
            dy = -0.018
        if m == 'v3':
            dy = -0.018
        if m == 'v4':
            dy = 0.008

        ax4.annotate(f"{arch_names_clean[i]}\n({p}M, AUC {auc_m:.3f})",
                     xy=(p, auc_m), xytext=(p * dx, auc_m + dy),
                     fontsize=9.5, fontweight='bold', color='#222222',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bbbbbb', alpha=0.9))

    ax4.set_xscale('log')
    ax4.set_xlim(1.5, 600)
    ax4.set_ylim(0.60, 0.85)
    ax4.set_title('(D) Fronteira de Eficiência: Parâmetros vs. AUC Externa de Melanoma', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Quantidade de Parâmetros (Milhões) — Escala Logarítmica', fontsize=11, fontweight='bold')
    ax4.set_ylabel('PAD-UFES-20 Melanoma AUC-ROC', fontsize=11, fontweight='bold')
    ax4.grid(True, which='both', linestyle='--', alpha=0.5)
    ax4.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

    plt.suptitle('Benchmarking Evolutivo MobileNet (V1–V5): Transferência Dermatoscópica e Clínica',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated high-legibility benchmark figure at: {out_path}")

if __name__ == '__main__':
    generate_figure()
