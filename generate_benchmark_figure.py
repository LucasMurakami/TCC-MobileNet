#!/usr/bin/env python3
"""
generate_benchmark_figure.py
Generates a publication-grade, high-legibility 6-panel benchmark comparison
for the TCC monograph, saving to:
- experiments/05_09_2026_run2/benchmark_comparison.png (4-panel benchmark comparison: A, B, C, D)
- experiments/05_09_2026_run2/operating_points_sweep.png (standalone operating points sweep)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def generate_figure():
    exp_dir = Path("experiments/05_09_2026_run2")
    csv_path = exp_dir / "master_leaderboard.csv"
    out_path = exp_dir / "benchmark_comparison.png"
    out_sweep_path = exp_dir / "operating_points_sweep.png"

    df = pd.read_csv(csv_path)

    # Filter main scenario runs and remove duplicates/backups
    main_df = df[(df['scenario'] == 'main') & (df['duplicate_of'].isna() | (df['duplicate_of'] == ''))].copy()

    # Define architectures and metadata
    archs = ['v1', 'v2', 'v3', 'v4', 'v5']
    arch_labels = ['V1\n(3.2M)', 'V2\n(2.2M)', 'V3\n(4.2M)', 'V4\n(8.4M)', 'V5\n(294.1M)']
    arch_names_clean = ['MobileNetV1', 'MobileNetV2', 'MobileNetV3', 'MobileNetV4', 'MobileNetV5']
    params_m = [3.2, 2.2, 4.2, 8.4, 294.1]
    arch_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    tau_keys = ['tau_0.50', 'tau_0.30', 'tau_0.20', 'tau_0.15', 'tau_0.10', 'tau_0.05', 'tau_0.02']
    tau_vals = np.array([0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02])

    # Compute stats across canonical seeds (42, 43, 44) for each architecture
    stats = {}
    for m in archs:
        m_runs = main_df[(main_df['model'] == m) & (main_df['seed'].isin([42, 43, 44]))]

        mel_recalls, bcc_recalls, akiec_recalls, total_recalls = [], [], [], []
        mel_op_sweep = {tv: [] for tv in tau_vals}
        mal_op_sweep = {tv: [] for tv in tau_vals}

        for s in [42, 43, 44]:
            p_dir = exp_dir / "scenarios" / "main" / m / f"seed{s}"
            cm_path = p_dir / "pad_ufes_20" / "confusion_gated.npy"
            if cm_path.exists():
                cm = np.load(cm_path)
                akiec_r = cm[0, [0, 1, 4]].sum() / cm[0].sum()
                bcc_r = cm[1, [0, 1, 4]].sum() / cm[1].sum()
                mel_r = cm[4, [0, 1, 4]].sum() / cm[4].sum()
                tot_r = (cm[0, [0, 1, 4]].sum() + cm[1, [0, 1, 4]].sum() + cm[4, [0, 1, 4]].sum()) / (cm[0].sum() + cm[1].sum() + cm[4].sum())
                mel_recalls.append(mel_r)
                bcc_recalls.append(bcc_r)
                akiec_recalls.append(akiec_r)
                total_recalls.append(tot_r)

            res_path = p_dir / "results.json"
            if res_path.exists():
                res_data = json.loads(res_path.read_text())
                for tk, tv in zip(tau_keys, tau_vals):
                    mel_pt = res_data.get('pad_mel_operating_points', {}).get(tk, {})
                    mal_pt = res_data.get('pad_malignant_operating_points', {}).get(tk, {})
                    mel_op_sweep[tv].append(mel_pt.get('sensitivity', np.nan))
                    mal_op_sweep[tv].append(mal_pt.get('sensitivity', np.nan))

        stats[m] = {
            'ham_mel_auc_mean': m_runs['ham_mel_auc_roc'].mean(),
            'ham_mel_auc_std': m_runs['ham_mel_auc_roc'].std(),
            'pad_mel_auc_mean': m_runs['mel_auc_roc'].mean(),
            'pad_mel_auc_std': m_runs['mel_auc_roc'].std(),
            'pad_macro_auc_mean': m_runs['pad_macro_auc_roc'].mean(),
            'pad_macro_auc_std': m_runs['pad_macro_auc_roc'].std(),

            'mel_recalls': mel_recalls,
            'mel_recall_mean': np.mean(mel_recalls),
            'mel_recall_std': np.std(mel_recalls, ddof=1) if len(mel_recalls) > 1 else 0.0,

            'bcc_recalls': bcc_recalls,
            'bcc_recall_mean': np.mean(bcc_recalls),
            'bcc_recall_std': np.std(bcc_recalls, ddof=1) if len(bcc_recalls) > 1 else 0.0,

            'akiec_recalls': akiec_recalls,
            'akiec_recall_mean': np.mean(akiec_recalls),
            'akiec_recall_std': np.std(akiec_recalls, ddof=1) if len(akiec_recalls) > 1 else 0.0,

            'total_recalls': total_recalls,
            'total_recall_mean': np.mean(total_recalls),
            'total_recall_std': np.std(total_recalls, ddof=1) if len(total_recalls) > 1 else 0.0,

            'ham_ece_mean': m_runs['ham_ece'].mean(),
            'ham_ece_std': m_runs['ham_ece'].std(),
            'pad_ece_mean': m_runs['pad_ece'].mean(),
            'pad_ece_std': m_runs['pad_ece'].std(),

            'mel_op_sweep': mel_op_sweep,
            'mal_op_sweep': mal_op_sweep,

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

    fig, axes = plt.subplots(2, 2, figsize=(17, 12.0))
    fig.patch.set_facecolor('#ffffff')

    x = np.arange(len(archs))
    width_2 = 0.36
    width_4 = 0.19

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
                        color='#1f77b4', edgecolor='#0f4c81', linewidth=1.2, alpha=0.92, zorder=3)
    bars1_pad = ax1.bar(x + width_2/2, pad_auc_means, width_2, yerr=pad_auc_stds,
                        capsize=4, label='Out-of-Domain: PAD-UFES-20 (Smartphone)',
                        color='#e6550d', edgecolor='#a63603', linewidth=1.2, alpha=0.92, zorder=3)

    for i, m in enumerate(archs):
        for idx_s, rec in enumerate(stats[m]['seeds']):
            offset = (idx_s - 1) * 0.045
            ax1.scatter(i - width_2/2 + offset, rec['ham_mel_auc_roc'],
                        color='#08306b', s=24, zorder=5, alpha=0.85, edgecolors='white', linewidth=0.5)
            ax1.scatter(i + width_2/2 + offset, rec['mel_auc_roc'],
                        color='#7f2704', s=24, zorder=5, alpha=0.85, edgecolors='white', linewidth=0.5)

    for i, bar in enumerate(bars1_ham):
        h = bar.get_height()
        s = ham_auc_stds[i] if not np.isnan(ham_auc_stds[i]) else 0.0
        seed_max = max([r['ham_mel_auc_roc'] for r in stats[archs[i]]['seeds']] + [h + s])
        y_pos = seed_max + 0.014
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#08306b',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    for i, bar in enumerate(bars1_pad):
        h = bar.get_height()
        s = pad_auc_stds[i] if not np.isnan(pad_auc_stds[i]) else 0.0
        seed_max = max([r['mel_auc_roc'] for r in stats[archs[i]]['seeds']] + [h + s])
        y_pos = seed_max + 0.014
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#a63603',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax1.set_title('(A) Degradação de Transferência: AUC-ROC de Melanoma', fontsize=12.5, fontweight='bold', pad=9)
    ax1.set_ylabel('AUC-ROC [0.50 - 1.00]', fontsize=10.5, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, fontsize=10.5, fontweight='bold')
    ax1.set_ylim(0.50, 1.10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 2 (Top-Right): PAD-UFES-20 Malignant Sensitivity Breakdown & Conjunct
    # ─────────────────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor('#fafafa')

    mel_means = [stats[m]['mel_recall_mean'] for m in archs]
    mel_stds = [stats[m]['mel_recall_std'] for m in archs]
    bcc_means = [stats[m]['bcc_recall_mean'] for m in archs]
    bcc_stds = [stats[m]['bcc_recall_std'] for m in archs]
    akiec_means = [stats[m]['akiec_recall_mean'] for m in archs]
    akiec_stds = [stats[m]['akiec_recall_std'] for m in archs]
    tot_means = [stats[m]['total_recall_mean'] for m in archs]
    tot_stds = [stats[m]['total_recall_std'] for m in archs]

    offsets = [-1.5 * width_4, -0.5 * width_4, 0.5 * width_4, 1.5 * width_4]

    bars2_mel = ax2.bar(x + offsets[0], mel_means, width_4, yerr=mel_stds,
                        capsize=3, label='Recall Melanoma (MEL)',
                        color='#e41a1c', edgecolor='#990000', linewidth=1.2, alpha=0.88, zorder=3)
    bars2_bcc = ax2.bar(x + offsets[1], bcc_means, width_4, yerr=bcc_stds,
                        capsize=3, label='Recall Basocelular (BCC)',
                        color='#377eb8', edgecolor='#184e77', linewidth=1.2, alpha=0.88, zorder=3)
    bars2_akiec = ax2.bar(x + offsets[2], akiec_means, width_4, yerr=akiec_stds,
                          capsize=3, label='Recall Queratose / CEC (AKIEC)',
                          color='#41b6c4', edgecolor='#238b45', linewidth=1.2, alpha=0.88, zorder=3)
    bars2_tot = ax2.bar(x + offsets[3], tot_means, width_4, yerr=tot_stds,
                        capsize=3, label='Sensibilidade Maligna Conjunta (Total)',
                        color='#756bb1', edgecolor='#3f007d', linewidth=1.3, alpha=0.92, zorder=3)

    bar_groups = [
        (bars2_mel, mel_means, mel_stds, '#990000', 'mel_recalls', '#67000d'),
        (bars2_bcc, bcc_means, bcc_stds, '#184e77', 'bcc_recalls', '#08306b'),
        (bars2_akiec, akiec_means, akiec_stds, '#238b45', 'akiec_recalls', '#00441b'),
        (bars2_tot, tot_means, tot_stds, '#3f007d', 'total_recalls', '#3f007d')
    ]

    for i in range(len(archs)):
        y_pos = []
        for g_idx, (bars, means, stds, _, key, dot_col) in enumerate(bar_groups):
            h = means[i]
            s = stds[i] if not np.isnan(stds[i]) else 0.0

            # Plot individual seeds dots on each bar with deterministic horizontal offset
            bar_center = bars[i].get_x() + bars[i].get_width()/2
            raw_seeds = stats[archs[i]][key]
            for s_idx, val in enumerate(raw_seeds):
                s_off = (s_idx - 1) * 0.038
                ax2.scatter(bar_center + s_off, val, color=dot_col, s=18, zorder=5,
                            alpha=0.9, edgecolors='white', linewidth=0.5)

            max_seed_val = max(raw_seeds + [h + s]) if raw_seeds else h + s
            y_pos.append(max_seed_val + 0.016)

        # Anti-crowding adjustment between adjacent bar labels
        for j in range(len(y_pos) - 1):
            if abs(y_pos[j+1] - y_pos[j]) < 0.045:
                if y_pos[j+1] >= y_pos[j]:
                    y_pos[j+1] += 0.025
                else:
                    y_pos[j] += 0.025

        for g_idx, (bars, means, _, col, _, _) in enumerate(bar_groups):
            bar = bars[i]
            h = means[i]
            ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos[g_idx]),
                         ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=col,
                         path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])

    ax2.set_title('(B) Sensibilidade no PAD-UFES-20: Classes Malignas e Triagem Conjunta (com Seeds)', fontsize=12.5, fontweight='bold', pad=9)
    ax2.set_ylabel('Sensibilidade / Recall [0.0 - 1.0]', fontsize=10.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, fontsize=10.5, fontweight='bold')
    ax2.set_ylim(0.0, 1.15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(loc='upper left', ncol=2, frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 3 (Middle-Left): Calibration Error (ECE)
    # ─────────────────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor('#fafafa')

    ham_ece_means = [stats[m]['ham_ece_mean'] for m in archs]
    ham_ece_stds = [stats[m]['ham_ece_std'] for m in archs]
    pad_ece_means = [stats[m]['pad_ece_mean'] for m in archs]
    pad_ece_stds = [stats[m]['pad_ece_std'] for m in archs]

    bars3_ham = ax3.bar(x - width_2/2, ham_ece_means, width_2, yerr=ham_ece_stds,
                        capsize=4, label='In-Domain ECE (HAM10000)',
                        color='#41b6c4', edgecolor='#225ea8', linewidth=1.2, alpha=0.9, zorder=3)
    bars3_pad = ax3.bar(x + width_2/2, pad_ece_means, width_2, yerr=pad_ece_stds,
                        capsize=4, label='Out-of-Domain ECE (PAD-UFES-20)',
                        color='#cb181d', edgecolor='#67000d', linewidth=1.2, alpha=0.9, zorder=3)

    for i, bar in enumerate(bars3_ham):
        h = bar.get_height()
        s = ham_ece_stds[i] if not np.isnan(ham_ece_stds[i]) else 0.0
        y_pos = h + s + 0.014
        ax3.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                     ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#225ea8',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    for i, bar in enumerate(bars3_pad):
        h = bar.get_height()
        s = pad_ece_stds[i] if not np.isnan(pad_ece_stds[i]) else 0.0
        y_pos = h + s + 0.014
        ax3.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                     ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#67000d',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax3.set_title('(C) Erro de Calibração Esperado (ECE) — Menor é Melhor', fontsize=12.5, fontweight='bold', pad=9)
    ax3.set_ylabel('ECE [0.0 - 0.6]', fontsize=10.5, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_labels, fontsize=10.5, fontweight='bold')
    ax3.set_ylim(0.0, 0.60)
    ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 4 (Middle-Right): Efficiency Trade-off: Parameters vs. PAD Melanoma AUC
    # ─────────────────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#fafafa')

    for i, m in enumerate(archs):
        p = params_m[i]
        auc_m = stats[m]['pad_mel_auc_mean']
        auc_s = stats[m]['pad_mel_auc_std']

        ax4.errorbar(p, auc_m, yerr=auc_s, fmt='o', color=arch_colors[i],
                     markersize=11, capsize=6, elinewidth=2, markeredgecolor='black', markeredgewidth=1.2,
                     zorder=4, label=f'{arch_names_clean[i]} ({p}M): {auc_m:.3f}')

    annotations_config = {
        'v1': {'text_pos': (2.0, 0.602),  'ha': 'center', 'va': 'top',    'rad': '0.0'},
        'v2': {'text_pos': (2.2, 0.775),  'ha': 'center', 'va': 'bottom', 'rad': '0.0'},
        'v3': {'text_pos': (6.2, 0.628),  'ha': 'left',   'va': 'center', 'rad': '0.0'},
        'v4': {'text_pos': (12.0, 0.755), 'ha': 'left',   'va': 'center', 'rad': '0.0'},
        'v5': {'text_pos': (160.0, 0.810),'ha': 'right',  'va': 'center', 'rad': '0.0'},
    }

    for i, m in enumerate(archs):
        p = params_m[i]
        auc_m = stats[m]['pad_mel_auc_mean']
        cfg = annotations_config[m]

        ax4.annotate(f"{arch_names_clean[i]}\n({p}M, AUC {auc_m:.3f})",
                     xy=(p, auc_m), xytext=cfg['text_pos'],
                     ha=cfg['ha'], va=cfg['va'],
                     fontsize=9.0, fontweight='bold', color='#1a1a1a',
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2, shrinkA=4, shrinkB=8,
                                     connectionstyle=f"arc3,rad={cfg['rad']}"),
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff',
                               edgecolor=arch_colors[i], linewidth=1.3, alpha=0.96),
                     zorder=6)

    ax4.set_xscale('log')
    ax4.set_xlim(0.85, 550)
    ax4.set_ylim(0.55, 0.86)
    ax4.set_title('(D) Fronteira de Eficiência: Parâmetros vs. AUC Externa de Melanoma', fontsize=12.5, fontweight='bold', pad=9)
    ax4.set_xlabel('Quantidade de Parâmetros (Milhões) — Escala Logarítmica', fontsize=10.5, fontweight='bold')
    ax4.set_ylabel('PAD-UFES-20 Melanoma AUC-ROC', fontsize=10.5, fontweight='bold', labelpad=12)
    ax4.grid(True, which='both', linestyle='--', alpha=0.5)
    ax4.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#bbbbbb', framealpha=0.95, fontsize=9.0)

    plt.suptitle('Benchmarking Evolutivo MobileNet (V1–V5): Transferência Dermatoscópica, Sensibilidade e Calibração',
                 fontsize=14.5, fontweight='bold', y=0.985)

    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.8, w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated 4-panel benchmark figure at: {out_path}")

    # Also generate the standalone 2-panel figure for operating points sweep
    fig_sweep, axes_sweep = plt.subplots(1, 2, figsize=(16, 6))
    fig_sweep.patch.set_facecolor('#ffffff')

    ax_s1, ax_s2 = axes_sweep[0], axes_sweep[1]
    ax_s1.set_facecolor('#fafafa')
    ax_s2.set_facecolor('#fafafa')

    # Melanoma curve
    for i, m in enumerate(archs):
        op_sweep = stats[m]['mel_op_sweep']
        means = [np.nanmean(op_sweep[tv]) for tv in tau_vals]
        stds = [np.nanstd(op_sweep[tv]) for tv in tau_vals]
        ax_s1.plot(tau_vals, means, marker='o', color=arch_colors[i], lw=2.2, markersize=7, label=f'{arch_names_clean[i]}', zorder=4)
        ax_s1.fill_between(tau_vals, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=arch_colors[i], alpha=0.12, zorder=2)
        for s_idx in range(len(stats[m]['seeds'])):
            seed_vals = [op_sweep[tv][s_idx] if s_idx < len(op_sweep[tv]) else np.nan for tv in tau_vals]
            ax_s1.plot(tau_vals, seed_vals, color=arch_colors[i], lw=0.8, linestyle=':', alpha=0.5, zorder=3)

    ax_s1.set_xlim(0.52, 0.0)
    ax_s1.set_ylim(0.0, 1.05)
    ax_s1.set_title('(A) Sensibilidade de Melanoma vs. Limiar Operacional $\\tau$', fontsize=12, fontweight='bold', pad=9)
    ax_s1.set_xlabel('Limiar Operacional $\\tau$ (Decrescente $\\rightarrow$ Triagem Mais Agressiva)', fontsize=10.5, fontweight='bold')
    ax_s1.set_ylabel('Sensibilidade de Melanoma (Recall)', fontsize=10.5, fontweight='bold')
    ax_s1.grid(True, linestyle='--', alpha=0.5)
    ax_s1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
    ax_s1.axvline(0.05, color='#e41a1c', linestyle='--', lw=1.2, alpha=0.7, zorder=1)
    ax_s1.annotate('Triagem Rastreio ($\\tau=0.05$)', xy=(0.05, 0.25), xytext=(0.075, 0.28),
                   fontsize=9, fontweight='bold', color='#990000',
                   arrowprops=dict(arrowstyle='->', color='#990000', lw=1.0),
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#e41a1c', alpha=0.9))

    # Malignant Joint curve
    for i, m in enumerate(archs):
        op_sweep = stats[m]['mal_op_sweep']
        means = [np.nanmean(op_sweep[tv]) for tv in tau_vals]
        stds = [np.nanstd(op_sweep[tv]) for tv in tau_vals]
        ax_s2.plot(tau_vals, means, marker='s', color=arch_colors[i], lw=2.2, markersize=7, label=f'{arch_names_clean[i]}', zorder=4)
        ax_s2.fill_between(tau_vals, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=arch_colors[i], alpha=0.12, zorder=2)
        for s_idx in range(len(stats[m]['seeds'])):
            seed_vals = [op_sweep[tv][s_idx] if s_idx < len(op_sweep[tv]) else np.nan for tv in tau_vals]
            ax_s2.plot(tau_vals, seed_vals, color=arch_colors[i], lw=0.8, linestyle=':', alpha=0.5, zorder=3)

    ax_s2.set_xlim(0.52, 0.0)
    ax_s2.set_ylim(0.15, 1.02)
    ax_s2.set_title('(B) Sensibilidade Maligna Conjunta vs. Limiar Operacional $\\tau$', fontsize=12, fontweight='bold', pad=9)
    ax_s2.set_xlabel('Limiar Operacional $\\tau$ (Decrescente $\\rightarrow$ Triagem Mais Agressiva)', fontsize=10.5, fontweight='bold')
    ax_s2.set_ylabel('Sensibilidade Maligna Conjunta (MEL+BCC+AKIEC)', fontsize=10.5, fontweight='bold')
    ax_s2.grid(True, linestyle='--', alpha=0.5)
    ax_s2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
    ax_s2.axhline(0.90, color='#238b45', linestyle='--', lw=1.2, alpha=0.7, zorder=1)
    ax_s2.annotate('Meta Clínica 90% Sensibilidade', xy=(0.35, 0.90), xytext=(0.42, 0.82),
                   fontsize=9, fontweight='bold', color='#238b45',
                   arrowprops=dict(arrowstyle='->', color='#238b45', lw=1.0),
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#238b45', alpha=0.9))

    plt.suptitle('Varredura de Limiares Operacionais de Triagem (Operating Points Sweep): $\\tau \\in [0.02, 0.50]$',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.5)
    plt.savefig(out_sweep_path, dpi=300, bbox_inches='tight')
    plt.close(fig_sweep)
    print(f"Successfully generated standalone sweep figure at: {out_sweep_path}")

if __name__ == '__main__':
    generate_figure()
