import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class CalibrationMetrics:
    @staticmethod
    def brier_score(y_true, y_prob):
        return np.mean((np.array(y_prob) - np.array(y_true)) ** 2)

    @staticmethod
    def skill_score(y_true, y_prob):
        b_model = CalibrationMetrics.brier_score(y_true, y_prob)
        p_base = np.mean(y_true)
        b_base = p_base * (1 - p_base)
        return (b_base - b_model) / b_base if b_base > 0 else 0.0

    @staticmethod
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        bin_bounds = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_true)
        
        if total == 0:
            return 0.0
            
        for i in range(n_bins):
            low, high = bin_bounds[i], bin_bounds[i + 1]
            in_bin = (y_prob >= low) & (y_prob < high)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                acc = np.mean(y_true[in_bin])
                conf = np.mean(y_prob[in_bin])
                ece += (bin_size / total) * abs(acc - conf)
                
        return ece


class ReliabilityPlotter:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _compute_bins(self, y_true, y_prob, n_bins=10):
        bins = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for p, label in zip(y_prob, y_true):
            bin_index = min(int(p * n_bins), n_bins - 1)
            bins[bin_index]["total"] += 1
            if label:
                bins[bin_index]["correct"] += 1
                
        return bins

    def _calculate_metrics(self, y_true, y_prob):
        return {
            "bref": np.mean(y_true) * (1 - np.mean(y_true)),
            "exact_match": np.mean(y_true),
            "brier": CalibrationMetrics.brier_score(y_true, y_prob),
            "skill": CalibrationMetrics.skill_score(y_true, y_prob),
            "ece": CalibrationMetrics.expected_calibration_error(y_true, y_prob)
        }

    def plot_reliability(self, y_true, y_prob, filename, title, rescaling):
        bins = self._compute_bins(y_true, y_prob)
        bin_centers = [(i + 0.5) / 10 for i in range(10)]
        p_correct = [
            (bins[i]["correct"] / bins[i]["total"]) if bins[i]["total"] > 0 else 0 
            for i in range(10)
        ]
        counts = [bins[i]["total"] for i in range(10)]
        metrics = self._calculate_metrics(y_true, y_prob)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(5, 6), sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )

        ax1.bar(bin_centers, p_correct, width=0.09, color='mediumpurple', edgecolor='black')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("P(correct)")
        ax1.set_title(title)

        metrics_text = (
            f"Rescaling: {rescaling}\n"
            f"Exact Match @1: {metrics['exact_match']*100:.0f}%\n"
            f"$B_{{\\mathrm{{ref}}}}$: {metrics['bref']:.2f}\n"
            f"ECE: {metrics['ece']:.2f}\n"
            f"$\\mathcal{{B}}$: {metrics['brier']:.2f}\n"
            f"SS: {metrics['skill']:.2f}"
        )
        
        ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=10)

        ax2.bar(bin_centers, counts, width=0.09, color='gray', edgecolor='black')
        ax2.set_ylabel("Count")
        ax2.set_xlabel("P(estimate)")
        ax2.set_xlim(0, 1)
        ax2.set_xticks(np.arange(0.0, 1.01, 0.1))

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved to: {filepath}")
        plt.close()


class ProbabilityCalibrator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def _prepare_data(self, calibrator_class, x_orig):
        return x_orig.reshape(-1, 1) if calibrator_class == LogisticRegression else x_orig

    def _get_predictions(self, calibrator, calibrator_class, x_test):
        if calibrator_class == LogisticRegression:
            return calibrator.predict_proba(x_test)[:, 1]
        else:
            return calibrator.transform(x_test.ravel())

    def calibrate_with_kfold(self, calibrator_class, x_orig, y_orig, **calibrator_params):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        calibrated_probs = np.zeros_like(x_orig, dtype=float)
        x_proc = self._prepare_data(calibrator_class, x_orig)

        for train_index, test_index in kf.split(x_proc, y_orig):
            x_train, x_test = x_proc[train_index], x_proc[test_index]
            y_train = y_orig[train_index]
            
            calibrator = calibrator_class(**calibrator_params)
            calibrator.fit(x_train, y_train)
            
            fold_probs = self._get_predictions(calibrator, calibrator_class, x_test)
            calibrated_probs[test_index] = fold_probs
            
        return calibrated_probs


class CalibrationAnalyzer:
    def __init__(self, data_path, output_dir="plots_kfold_original_style"):
        self.data_path = data_path
        self.plotter = ReliabilityPlotter(output_dir)
        self.calibrator = ProbabilityCalibrator()
        
    def load_data(self):
        try:
            with open(self.data_path) as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.data_path}' not found.")
            
        y_true = np.array([int(sample["is_correct"]) for sample in json_data])
        y_prob_orig = np.array([sample["pavg"] for sample in json_data])
        
        print(f"Total samples loaded: {len(y_true)}\n")
        return y_true, y_prob_orig

    def run_analysis(self):
        y_true, y_prob_orig = self.load_data()
        
        base_title = "DyPyBench deepseek-coder-1.3b Reliability Plot"
        
        print("Generating plot for uncalibrated data...")
        self.plotter.plot_reliability(
            y_true, y_prob_orig,
            "reliability_uncalibrated.png",
            base_title, "None"
        )

        print("Applying Isotonic Regression with 5-fold cross-validation...")
        iso_params = {'out_of_bounds': 'clip', 'y_min': 0, 'y_max': 1}
        y_prob_iso = self.calibrator.calibrate_with_kfold(
            IsotonicRegression, y_prob_orig, y_true, **iso_params
        )
        self.plotter.plot_reliability(
            y_true, y_prob_iso,
            "reliability_isotonic_kfold.png",
            base_title, "Isotonic"
        )

        print("Applying Platt Scaling with 5-fold cross-validation...")
        y_prob_platt = self.calibrator.calibrate_with_kfold(
            LogisticRegression, y_prob_orig, y_true
        )
        self.plotter.plot_reliability(
            y_true, y_prob_platt,
            "reliability_platt_kfold.png",
            base_title, "Platt"
        )

        print(f"\nProcess completed. Check the folder '{self.plotter.output_dir}'.")


def main():
    analyzer = CalibrationAnalyzer("results/dypybench_predictions_deepseek.json")
    analyzer.run_analysis()


if __name__ == "__main__":
    main()