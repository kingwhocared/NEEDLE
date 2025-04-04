import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.MyOpenAIUtils import GPT_MODEL

# Load data
file_path = "df.csv"
df = pd.read_csv(file_path)

# Filter UMWP inconsistencies
df = df[~((df["dataset_source"] == "UMWP") & (df["ground_truth_answer"] != "UNANSWERABLE"))]

def classify_naked(row):
    if row["dataset_source"] == "UMWP":
        if row["ground_truth_answer"] == "UNANSWERABLE":
            if row["proposed_answer"] == "COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION":
                return "Correctly didn't hallucinate an answer"
            else:
                return "Hallucinated an answer"
    if row["proposed_answer"] == "COULD_NOT_PROVIDE_ANSWER":
        return "COULD_NOT_PROVIDE_ANSWER"
    else:
        prop = float(row["proposed_answer"])
        truth = float(row["ground_truth_answer"])
        return "Correct" if prop == truth else "Wrong"

def classify_needle(row):
    if row["dataset_source"] in ["GSM8K", "CIAR"]:
        try:
            prop = float(row["proposed_answer"])
            truth = float(row["ground_truth_answer"])
            return "Correct" if prop == truth else "Wrong"
        except:
            ret = row["proposed_answer"]
            if ret == "UNCERTAIN_SOLUTION":
                return "Rejected due to uncertainty"
            elif ret == "UNANSWERABLE":
                return "Rejected input as invalid"
            return ret
    elif row["dataset_source"] == "UMWP":
        if row["proposed_answer"] == "UNANSWERABLE":
            return "Correctly rejected input as unanswerable" if row["ground_truth_answer"] == "UNANSWERABLE" else "Falsely claiming to be unanswerable"
        elif row["proposed_answer"] == "UNCERTAIN_SOLUTION":
            return "Rejected due to uncertainty"
        elif row["ground_truth_answer"] == "UNANSWERABLE":
            return "Hallucinated an answer"
        else:
            try:
                prop = float(row["proposed_answer"])
                truth = float(row["ground_truth_answer"])
                return "Correct" if prop == truth else "Wrong"
            except:
                return row["proposed_answer"]
    else:
        raise RuntimeError

# Classify results
df.loc[df["model"] == "NAKED_GPT", "result"] = df[df["model"] == "NAKED_GPT"].apply(classify_naked, axis=1)
df.loc[df["model"] == "NEEDLE", "result"] = df[df["model"] == "NEEDLE"].apply(classify_needle, axis=1)

# Setup
dpi = 600
models = ["NAKED_GPT", "NEEDLE"]
datasets = ["GSM8K", "CIAR", "UMWP"]

categories = [
    'Wrong',
    'Hallucinated an answer',
    'Falsely claiming to be unanswerable',
    'Rejected due to uncertainty',
    'Rejected input as invalid',
    'Correct',
    'Correctly rejected input as unanswerable',
    "Correctly didn't hallucinate an answer"
]

color_map = {
    "Correct": "darkgreen",
    "Wrong": "darkred",
    "Hallucinated an answer": "darkred",
    "Correctly didn't hallucinate an answer": "darkgreen",
    "Rejected due to uncertainty": "gray",
    "Rejected input as invalid": "darkgray",
    "Correctly rejected input as unanswerable": "darkgreen",
    "Falsely claiming to be unanswerable": "darkorange",
}

# Generate stacked bar chart for each dataset
for dataset in datasets:
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(models))
    width = 0.6
    bottoms = [0] * len(models)
    used_categories = []

    bar_segments = {}  # Store bar info for arrow reference

    for cat in categories:
        values = []
        for model in models:
            subset = df[(df["model"] == model) & (df["dataset_source"] == dataset)]
            total = len(subset)
            count = subset["result"].value_counts().get(cat, 0)
            percentage = 100 * count / total if total > 0 else 0
            values.append(percentage)

        if any(v > 0 for v in values):
            used_categories.append(cat)
            bars = ax.bar(x, values, width, bottom=bottoms, color=color_map.get(cat, "purple"), label=cat)

            if cat in {"Wrong", "Hallucinated an answer"}:
                # Save coordinates of Wrong bars for annotation
                for i, bar in enumerate(bars):
                    bar_segments[f"{i}"] = {
                        "x": bar.get_x(),
                        "width": bar.get_width(),
                        "y": bar.get_y(),
                        "height": bar.get_height()
                    }

            for bar, val in zip(bars, values):
                if val > 2:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%",
                        ha='center', va='center', color='white',
                        fontsize=12, fontweight='bold'
                    )
            bottoms = [b + v for b, v in zip(bottoms, values)]

        # Add arrow between "Wrong" bars + annotation
        if "0" in bar_segments and "1" in bar_segments:
            bar0 = bar_segments["0"]
            bar1 = bar_segments["1"]

            val0 = bar0["height"]
            val1 = bar1["height"]
            diff = 1 - (val1 / val0)
            diff *= 100
            if diff > 0:
                percentage_change_text = f"{diff:.1f}%"

                start = (bar0["x"] + bar0["width"], bar0["y"] + bar0["height"])
                end = (bar1["x"], bar1["y"] + bar1["height"])

                ax.annotate(
                    '', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    annotation_clip=False
                )

                # Calculate mid-point for annotation
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2

                ax.text(
                    mid_x, mid_y,
                    percentage_change_text,
                    ha='center', va='bottom', backgroundcolor=(1, 1, 1, 0.5),
                    fontsize=12, fontweight='bold', color='darkgreen'
                )

    # Axis labels
    model_labels = [GPT_MODEL if m == "NAKED_GPT" else "NEEDLE (gpt-4o-mini)" for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.set_ylabel("Percentage", fontsize=13)
    ax.set_title(f"{dataset}", fontsize=15)
    ax.set_ylim(0, 100)

    # Reverse legend (top-down order matches stacked bars)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in reversed(categories) if c in used_categories
    ]
    legend_labels = [c for c in reversed(categories) if c in used_categories]
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)

    plt.tight_layout()
    fig.savefig(f"eval_{dataset}_pct_annotated.svg", format="svg", dpi=dpi)
    fig.savefig(f"eval_{dataset}_pct_annotated.png", format="png", dpi=dpi)
    plt.show()
