"""Build the review notebook from generated analysis artifacts."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_secom_yield_and_defect_prediction.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # Semiconductor Manufacturing Yield and Defect Prediction

        This notebook is the review layer for a leakage-aware UCI SECOM analysis.
        The reusable implementation lives in `src/` and the two run scripts. It
        reports favourable and unfavourable results, including the failure of a
        random-split threshold to transfer to a later process period.
        """
    ),
    code(
        """
        from pathlib import Path
        import pandas as pd
        from IPython.display import Image, display

        ROOT = Path('..').resolve()
        REPORTS = ROOT / 'reports'
        TABLES = REPORTS / 'tables'
        FIGURES = REPORTS / 'figures'
        """
    ),
    markdown(
        """
        ## 1. Data audit and time variation

        SECOM contains 104 failures among 1,567 runs and 590 anonymised process
        measurements. Failure prevalence is not stable across the collection
        period, which motivates chronological evaluation rather than random
        splitting alone.
        """
    ),
    code(
        """
        profile = pd.read_csv(TABLES / 'dataset_profile.csv')
        display(profile.T.rename(columns={0: 'value'}))
        display(Image(filename=str(FIGURES / 'weekly_failure_rate.png'), width=900))
        """
    ),
    markdown(
        """
        ## 2. Leakage-safe model comparison

        Missingness filtering, imputation, constant-feature removal, scaling,
        feature selection, PCA, and missingness indicators are fitted inside each
        training fold. Models within 0.005 mean PR-AUC are treated as practical
        ties to avoid selecting complexity from a negligible decimal difference.
        """
    ),
    code(
        """
        comparison = pd.read_csv(TABLES / 'model_cv_comparison.csv')
        comparison[['model', 'cv_pr_auc_mean', 'cv_pr_auc_std',
                    'cv_roc_auc_mean', 'cv_recall_mean']].round(3)
        """
    ),
    markdown(
        """
        ## 3. Random versus chronological holdout

        Random splitting gives ROC-AUC 0.739 and PR-AUC 0.244. Training on the
        earliest 80% and testing on the latest 20% reduces these to 0.616 and
        0.087. The chronological result is the more deployment-like stress test.
        """
    ),
    code(
        """
        validation = pd.read_csv(TABLES / 'random_vs_chronological_metrics.csv')
        display(validation[['validation_scheme', 'roc_auc', 'pr_auc', 'recall',
                            'precision', 'review_rate']].round(3))
        display(Image(filename=str(FIGURES / 'random_vs_chronological_metrics.png'), width=950))
        """
    ),
    markdown("## 4. Bootstrap uncertainty"),
    code(
        """
        intervals = pd.read_csv(TABLES / 'bootstrap_confidence_intervals.csv')
        intervals[['validation_scheme', 'metric', 'estimate', 'lower', 'upper']].round(3)
        """
    ),
    markdown(
        """
        ## 5. Review-capacity analysis

        A screening team usually has a review budget, not a universal probability
        threshold. The chronological top 10% captures 17.6% of failures (1.73x
        lift), and the top 20% captures 35.3%.
        """
    ),
    code(
        """
        capacity = pd.read_csv(TABLES / 'review_capacity_analysis.csv')
        display(capacity[['validation_scheme', 'target_review_fraction',
                          'failures_captured', 'total_failures', 'failure_recall',
                          'review_precision', 'lift_vs_baseline']].round(3))
        display(Image(filename=str(FIGURES / 'review_capacity_curve.png'), width=850))
        """
    ),
    markdown(
        """
        ## 6. Cost-ratio sensitivity

        Thresholds are derived from non-test predictions. The policy is highly
        sensitive to the assumed missed-failure cost, and the 10:1 temporal policy
        detects none of the latest 17 failures. No threshold is promoted as a
        production rule.
        """
    ),
    code(
        """
        cost = pd.read_csv(TABLES / 'cost_ratio_sensitivity.csv')
        cost[['validation_scheme', 'fn_fp_cost_ratio', 'threshold', 'recall',
              'precision', 'false_positive', 'false_negative', 'review_rate']].round(3)
        """
    ),
    markdown(
        """
        ## 7. Probability calibration

        Isotonic calibration is best by Brier score and five-bin calibration error,
        while sigmoid calibration is best by log loss. The disagreement and small
        holdout argue against promoting either probability scale.
        """
    ),
    code(
        """
        calibration = pd.read_csv(TABLES / 'calibration_comparison.csv')
        display(calibration.round(4))
        display(Image(filename=str(FIGURES / 'calibration_comparison.png'), width=750))
        """
    ),
    markdown(
        """
        ## 8. Feature-ranking stability

        `sensor_103` and `sensor_059` rank in the model's top 20 in all 10 folds.
        Held-out permutation importance independently ranks `sensor_059` first.
        Anonymous variable identities prevent physical or causal interpretation.
        """
    ),
    code(
        """
        stability = pd.read_csv(TABLES / 'feature_stability.csv')
        display(stability.head(15).round(3))
        display(Image(filename=str(FIGURES / 'feature_stability.png'), width=800))
        """
    ),
    markdown(
        """
        ## 9. Reproduction and limits

        Run `python scripts/run_baseline.py` and then
        `python scripts/run_enhanced_analysis.py` from the project root. The model
        card records intended use and limitations. This is a portfolio screening
        workflow, not a deployable fab decision system.
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
