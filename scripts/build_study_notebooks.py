"""Build and execute three review notebooks using the invoking interpreter."""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'reports/study'
summary = json.loads((OUT / 'summary.json').read_text())
def table(name): return pd.read_csv(OUT / f'{name}.csv')
"""

NOTEBOOKS = [
    ("01_data_and_question.ipynb", [
        ("m", "# 1. Data quality and the research question\nCan a historical manufacturing risk model remain useful in later periods? This is an AI-assisted, independent MSc portfolio study using public anonymous data, not a course submission or industrial engagement. All periods were previously inspected; later results are retrospective. Independent author mastery is not established by these executed cells."),
        ("c", SETUP),
        ("m", "## Audit the actual input\nLabels represent recorded pass/fail outcomes. Sensor identity, units, equipment and specifications are unavailable. Anonymous variables cannot be assigned a physical mechanism."),
        ("c", "from secom_project.data import load_secom\nX, y, timestamps = load_secom(ROOT / 'data/raw')\ndisplay(pd.Series({'rows': len(X), 'variables': X.shape[1], 'failures': int(y.sum()), 'missing_cells': int(X.isna().sum().sum()), 'duplicate_timestamps': int(timestamps.duplicated().sum())}).to_frame('value'))\nassert len(X) == summary['data']['rows']\ndisplay(table('data_quality').sort_values('missing_rate', ascending=False).head(10))"),
        ("m", "## Why time is part of the question\nFailure proportions and sample sizes change across time. This does not by itself prove concept drift or a physical cause. Equal timestamps are grouped at each split; unknown lot dependence remains unresolved."),
        ("c", "weekly = table('weekly_failure_rate')\ndisplay(weekly)\ndisplay(table('splits'))"),
        ("m", "## Provenance\nThe manifest records raw-data hashes, source hashes, package versions and the actual configuration. Its git HEAD is the base revision at generation; source hashes identify any uncommitted implementation used."),
        ("c", "import hashlib\nmanifest = json.loads((OUT / 'manifest.json').read_text())\nfor name, expected in manifest['data_sha256'].items():\n    assert hashlib.sha256((ROOT/'data/raw'/name).read_bytes()).hexdigest() == expected\ndisplay(pd.Series(manifest['package_versions']).to_frame('version'))"),
    ]),
    ("02_time_validation_and_review.ipynb", [
        ("m", "# 2. Time validation and limited review capacity\nFour outer historical periods. Logistic regression and random forest compete using only earlier inner windows. Constant risk is the baseline; feature-selection and missingness-indicator ablations are fixed comparisons, never inputs to policy selection."),
        ("c", SETUP),
        ("c", "display(pd.DataFrame(summary['decisions']))\ndisplay(table('inner_validation'))\ndisplay(table('temporal_metrics'))"),
        ("m", "## Recompute rather than copy a headline\nAverage precision measures ranking. Class-weighted scores are not calibrated failure probabilities. A model can improve ranking while having a much worse Brier score than constant prevalence."),
        ("c", "from secom_project.study import score_metrics, capacity_evidence\npred = table('predictions')\nlast = pred[pred.window == pred.window.max()]\nmetrics = score_metrics(last.truth, last.risk)\nassert np.isclose(metrics['average_precision'], summary['latest_metrics']['average_precision'])\ndisplay(pd.Series(metrics).to_frame('recomputed'))\ndisplay(capacity_evidence(last.truth, last.risk, [.05,.1,.2,.3]))"),
        ("c", "display(Image(filename=str(OUT/'figures/study_overview.png'), width=1000))\ndisplay(table('random_split_diagnostic'))"),
        ("m", "## What uncertainty means here\nWhole-day bootstrap holds the model fixed and resamples just 16 observed days in the latest period. It is not an external-validation guarantee. Random-review intervals describe a separate hypergeometric null, not model-confidence intervals. Report all capacities; do not select the most flattering result."),
        ("c", "display(table('conditional_day_bootstrap'))\ndisplay(table('illustrative_policy_cost'))"),
        ("m", "## Cases selected by a fixed rule\nThe earliest captured failure, missed failure and false alert at 20% batch capacity are shown. This batch policy requires the complete batch; the past-validation percentile is separately frozen and does not update past records when new records arrive."),
        ("c", "display(table('case_studies'))"),
    ]),
    ("03_alarm_diagnosis_and_simulation.ipynb", [
        ("m", "# 3. Why alarms can fail to identify failures\nA process excursion and a recorded failure are different events. Monitoring variables are now selected within the early reference window. Historically labelled pass runs approximate normal operation, but do not establish an in-control process."),
        ("c", SETUP),
        ("c", "display(table('control_limits'))\ndisplay(table('monitor_methods'))\ndisplay(table('alarm_union_diagnostics'))"),
        ("m", "## Diagnose the v2 discoveries\nSensors 542 and 543 were chosen after examining v2. They are post-hoc examples, not independent discoveries. Inspect shifts among both passed and failed records, measurement resolution, missingness and actual fitted limits. The evidence cannot identify the physical cause of the shift."),
        ("c", "d = table('sensor_diagnostics')\ndisplay(d[d.feature.isin(['sensor_542','sensor_543'])])\ndisplay(Image(filename=str(OUT/'figures/alarm_diagnosis.png'), width=1000))"),
        ("m", "## Reference sensitivity\nVariables are kept fixed while the reference population changes. All variants are reported; no later-label winner is promoted. Sensitivity variants start state at the evaluation boundary, whereas the main chart carries state from the end of reference fitting."),
        ("c", "display(table('reference_sensitivity'))"),
        ("m", "## Known-change simulation (separate synthetic experiment)\nUse stable Gaussian noise, a 1-sigma mean step and a gradual rise to 2 sigma. Repeated trials estimate pre-change false alarms and post-change detection. Undetected changes are retained as censored trials. No real fabrication lead time is inferred."),
        ("c", "display(Image(filename=str(OUT/'figures/simulation.png'), width=1000))\ndisplay(table('simulation_summary'))\ntrials = table('simulation_trials')\nassert trials.groupby(['scenario','method']).size().nunique() == 1"),
        ("m", "## Explanation boundaries\nQ/SPE residual contributions explain PCA reconstruction error only. Classifier permutation importance is evaluated in an earlier validation block and is a noisy association diagnostic. Neither identifies a causal physical root cause."),
        ("c", "display(table('earlier_validation_importance').head(10))\ndisplay(table('case_studies')[['run_id','case_type','q_residual_top3_only']])"),
    ]),
]


def main():
    for filename, content in NOTEBOOKS:
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(text) if kind == 'm' else nbf.v4.new_code_cell(text) for kind,text in content]
        nb.metadata.kernelspec = {"name":"python3", "display_name":"Python 3", "language":"python"}
        client = NotebookClient(nb, timeout=120, resources={"metadata":{"path":str(ROOT)}})
        client.create_kernel_manager()
        client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.execute()
        nbf.write(nb, ROOT / 'notebooks' / filename)
        print(f"Executed {filename}", flush=True)


if __name__ == '__main__':
    main()
