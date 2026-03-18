import matplotlib
matplotlib.use("Agg")

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _coerce_target(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )
        if mapped.notna().sum() == len(series):
            return mapped.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def _save_plot(fig, output_path: str) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_target_distribution(df: pd.DataFrame, target_col: str, output_dir: str) -> str:
    counts = df[target_col].value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")

    output_file = "churn_distribution.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_missing_values(df: pd.DataFrame, output_dir: str) -> str:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(missing) > 0:
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=12)
        ax.set_axis_off()

    output_file = "missing_values.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_numerical_histograms(df: pd.DataFrame, numeric_cols: List[str], output_dir: str) -> str:
    cols = [c for c in numeric_cols if c != "__target__"][:6]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(cols):
            df[cols[i]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title(cols[i])
        else:
            ax.set_axis_off()

    output_file = "numerical_histograms.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], output_dir: str) -> str:
    cols = [c for c in numeric_cols if c != "__target__"][:10]

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(cols) >= 2:
        corr = df[cols].corr(numeric_only=True)
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation heatmap", ha="center", va="center")
        ax.set_axis_off()

    output_file = "correlation_heatmap.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_tenure_vs_churn(df: pd.DataFrame, tenure_col: Optional[str], output_dir: str) -> Optional[str]:
    if not tenure_col:
        return None

    grp = df.groupby(tenure_col)["__target__"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    grp.plot(ax=ax)
    ax.set_title("Churn Rate vs Tenure")
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Average Churn Rate")

    output_file = "churn_vs_tenure.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_monthly_charges_vs_churn(df: pd.DataFrame, monthly_col: Optional[str], output_dir: str) -> Optional[str]:
    if not monthly_col:
        return None

    temp = df[[monthly_col, "__target__"]].dropna().copy()

    if temp.empty:
        return None

    temp["charge_bin"] = pd.qcut(temp[monthly_col], q=10, duplicates="drop")
    grp = temp.groupby("charge_bin", observed=False)["__target__"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    grp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Monthly Charges Bin")
    ax.set_xlabel("Monthly Charges Bin")
    ax.set_ylabel("Average Churn Rate")

    output_file = "churn_vs_monthly_charges.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_contract_vs_churn(df: pd.DataFrame, contract_col: Optional[str], output_dir: str) -> Optional[str]:
    if not contract_col:
        return None

    grp = (
        df.groupby(contract_col)["__target__"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    grp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Average Churn Rate")

    output_file = "contract_vs_churn.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def _plot_boxplots_by_churn(
    df: pd.DataFrame,
    candidate_cols: List[str],
    output_dir: str
) -> Optional[str]:
    cols = [c for c in candidate_cols if c in df.columns][:3]

    if not cols:
        return None

    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        churn_0 = df[df["__target__"] == 0][col].dropna()
        churn_1 = df[df["__target__"] == 1][col].dropna()

        ax.boxplot([churn_0, churn_1], labels=["No Churn", "Churn"])
        ax.set_title(f"{col} by Churn")

    output_file = "boxplots_by_churn.png"
    _save_plot(fig, os.path.join(output_dir, output_file))
    return output_file


def run_eda(file_path: str, output_dir: str = "eda_outputs") -> Dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)
        # Normalize common numeric columns
    for col_name in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        real_col = _find_column(df, [col_name])
        if real_col:
            df[real_col] = pd.to_numeric(df[real_col], errors="coerce")

    target_col = _find_column(df, ["churn", "target", "label", "exited", "attrition"])
    if not target_col:
        raise ValueError("Could not find target column. Expected one of: churn, target, label, exited, attrition")

    df["__target__"] = _coerce_target(df[target_col])
    df = df.dropna(subset=["__target__"]).copy()
    df["__target__"] = df["__target__"].astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    tenure_col = _find_column(df, ["tenure"])
    monthly_col = _find_column(df, ["monthly_charges", "monthlycharges"])
    total_col = _find_column(df, ["total_charges", "totalcharges"])
    contract_col = _find_column(df, ["contract"])

    generated_files = []

    generated_files.append(_plot_target_distribution(df, target_col, output_dir))
    generated_files.append(_plot_missing_values(df, output_dir))
    generated_files.append(_plot_numerical_histograms(df, numeric_cols, output_dir))
    generated_files.append(_plot_correlation_heatmap(df, numeric_cols, output_dir))

    for maybe_file in [
        _plot_tenure_vs_churn(df, tenure_col, output_dir),
        _plot_monthly_charges_vs_churn(df, monthly_col, output_dir),
        _plot_contract_vs_churn(df, contract_col, output_dir),
        _plot_boxplots_by_churn(df, [tenure_col, monthly_col, total_col], output_dir),
    ]:
        if maybe_file:
            generated_files.append(maybe_file)

    churn_counts = df["__target__"].value_counts().to_dict()
    churn_rate = float(df["__target__"].mean())

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False).to_dict()

    summary = {
        "file_path": file_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_column": target_col,
        "churn_distribution": {str(k): int(v) for k, v in churn_counts.items()},
        "churn_rate": round(churn_rate, 4),
        "numeric_columns": [c for c in numeric_cols if c != "__target__"],
        "missing_values": missing_summary,
        "plots": generated_files,
    }

    if contract_col:
        contract_stats = (
            df.groupby(contract_col)["__target__"]
            .mean()
            .sort_values(ascending=False)
            .round(4)
            .to_dict()
        )
        summary["churn_rate_by_contract"] = {str(k): float(v) for k, v in contract_stats.items()}

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    result = run_eda("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", "eda_outputs")
    print(json.dumps(result, indent=2))