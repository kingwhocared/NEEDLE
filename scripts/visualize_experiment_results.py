import pandas as pd
import matplotlib.pyplot as plt

from utils.MyOpenAIUtils import GPT_MODEL

# Load the CSV file
file_path = "df.csv"
df = pd.read_csv(file_path)

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
                ret = "Rejected due to uncertainty"
            elif ret == "UNANSWERABLE":
                ret = "Rejected input as invalid"
            return ret
    elif row["dataset_source"] == "UMWP":
        if row["proposed_answer"] == "UNANSWERABLE":
            return "Correctly rejected input as unanswerable" if row[
                                                                     "ground_truth_answer"] == "UNANSWERABLE" else "Falsely claiming to be unanswerable"
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


df.loc[df["model"] == "NAKED_GPT", "result"] = df[df["model"] == "NAKED_GPT"].apply(classify_naked, axis=1)
df.loc[df["model"] == "NEEDLE", "result"] = df[df["model"] == "NEEDLE"].apply(classify_needle, axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
models = ["NAKED_GPT", "NEEDLE"]
datasets = ["GSM8K", "CIAR", "UMWP"]

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

for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        subset = df[(df["model"] == model) & (df["dataset_source"] == dataset)]
        counts = subset["result"].value_counts().reindex([
            'Wrong',
            'Hallucinated an answer',
            'Falsely claiming to be unanswerable',

            'Rejected due to uncertainty',
            'Rejected input as invalid',

            'Correct',
            'Correctly rejected input as unanswerable',
            "Correctly didn't hallucinate an answer",
        ]).dropna()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()

        # Pick colors for each slice
        colors = [color_map.get(label, "purple") for label in labels]

        # Create pie without slice labels
        wedges, _ = ax.pie(sizes, startangle=90, colors=colors)

        # Build a custom legend that shows label + percentage
        total = sum(sizes)
        legend_labels = [f"{lbl} ({val / total * 100:.1f}%)" for lbl, val in zip(labels, sizes)]
        ax.legend(wedges, legend_labels, loc="center", bbox_to_anchor=(0.5, 0.5), frameon=True,
                  facecolor="white").get_frame().set_alpha(1)

        mode_display_name = model
        if model == "NAKED_GPT":
            mode_display_name = GPT_MODEL
        ax.set_title(f"{dataset} | {mode_display_name}")

plt.tight_layout()
fig.savefig('myimage.svg', format='svg', dpi=1200)
