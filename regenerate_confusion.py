import argparse
import json
from pathlib import Path

import numpy as np

from dataset import CLASS_NAMES
from metrics import (
    confusion_summary,
    decide_argmax,
    decide_malignant_gated,
    decide_prior_corrected,
    select_logit_adjust,
)
from visualize import plot_confusion_matrices, plot_decision_confusion_matrices


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate confusion matrices from saved prediction artifacts")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--scenario", default=None)
    return parser.parse_args()


def serializable_summary(summary):
    clinical = summary["malignant"]
    return {
        "accuracy": summary["accuracy"],
        "balanced_accuracy": summary["balanced_accuracy"],
        "macro_f1": summary["macro_f1"],
        "malignant_sensitivity": clinical["sensitivity"],
        "malignant_specificity": clinical["specificity"],
        "malignant_ppv": clinical["ppv"],
        "malignant_npv": clinical["npv"],
    }


def decision_metrics(probs, targets, priors, tau, malignant_threshold):
    malignant = [CLASS_NAMES.index(name) for name in ("akiec", "bcc", "mel")]
    predictions = {
        "argmax": decide_argmax(probs),
        "prior_corrected": decide_prior_corrected(probs, priors, tau),
        "malignant_gated": decide_malignant_gated(probs, malignant_threshold, malignant),
    }
    return {
        name: serializable_summary(confusion_summary(targets, pred, CLASS_NAMES, malignant))
        for name, pred in predictions.items()
    }, predictions


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()
    with open(session_dir / "class_priors.json", encoding="utf-8") as file:
        prior_map = json.load(file)
    priors = np.asarray([prior_map[name] for name in CLASS_NAMES], dtype=float)
    result_paths = sorted(session_dir.glob("scenarios/**/results.json"))
    processed = 0

    for result_path in result_paths:
        model_dir = result_path.parent
        model_name = model_dir.name if not model_dir.name.startswith("seed") else model_dir.parent.name
        scenario_name = model_dir.relative_to(session_dir / "scenarios").parts[0]
        if args.model and model_name != args.model:
            continue
        if args.scenario and scenario_name != args.scenario:
            continue
        ham_dir = model_dir / "ham10000"
        pad_dir = model_dir / "pad_ufes_20"
        required = [ham_dir / "all_probs.npy", ham_dir / "all_targets.npy", pad_dir / "all_probs.npy", pad_dir / "all_targets.npy"]
        if not all(path.exists() for path in required):
            print(f"Skipping {scenario_name}/{model_name}: prediction artifacts missing")
            continue

        with open(result_path, encoding="utf-8") as file:
            results = json.load(file)
        ham_probs = np.load(required[0])
        ham_targets = np.load(required[1])
        tau, tau_table = select_logit_adjust(ham_probs, ham_targets, priors)
        malignant_threshold = float(results.get("malignant_triage_threshold", 0.25))
        output = {
            "model": model_name,
            "scenario": scenario_name,
            "tau_star": tau,
            "tau_source": "ham_test (05_09 backfill)",
            "tau_sweep": tau_table,
            "malignant_threshold": malignant_threshold,
            "domains": {},
        }

        for domain_name, domain_dir in (("ham10000", ham_dir), ("pad_ufes_20", pad_dir)):
            probs = np.load(domain_dir / "all_probs.npy")
            targets = np.load(domain_dir / "all_targets.npy")
            metrics, predictions = decision_metrics(probs, targets, priors, tau, malignant_threshold)
            output["domains"][domain_name] = metrics
            np.save(domain_dir / "confusion_argmax.npy", confusion_summary(targets, predictions["argmax"], CLASS_NAMES)["counts"])
            np.save(domain_dir / "confusion_tau.npy", confusion_summary(targets, predictions["prior_corrected"], CLASS_NAMES)["counts"])
            np.save(domain_dir / "confusion_gated.npy", confusion_summary(targets, predictions["malignant_gated"], CLASS_NAMES)["counts"])
            plot_confusion_matrices(targets, predictions["argmax"], CLASS_NAMES, domain_dir / "confusion_matrix.png", f"{model_name} ({domain_name})")
            plot_decision_confusion_matrices(probs, targets, CLASS_NAMES, priors, tau, malignant_threshold,
                                             domain_dir / "confusion_matrix_decision.png", f"{model_name} ({domain_name})")

        with open(model_dir / "decision_metrics.json", "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        processed += 1
        print(f"Regenerated {scenario_name}/{model_name}: tau={tau:.1f}")

    print(f"Completed {processed} run(s)")


if __name__ == "__main__":
    main()
