import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "df.csv"
df = pd.read_csv(file_path)

# Define function for NAKED_GPT classification
def classify_naked(row):
    if row["dataset_source"] == "UMWP":
        if (row["proposed_answer"] == "COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION" and
            row["ground_truth_answer"] == "UNANSWERABLE"):
            return "Correct"
    if row["proposed_answer"] == row["ground_truth_answer"]:
        return "Correct"
    elif row["proposed_answer"] == "COULD_NOT_PROVIDE_ANSWER":
        return "COULD_NOT_PROVIDE_ANSWER"
    else:
        return "Wrong"

# Define function for NEEDLE classification
def classify_needle(row):
    if row["dataset_source"] in ["GSM8K", "CIAR"]:
        try:
            prop = float(row["proposed_answer"])
            truth = float(row["ground_truth_answer"])
            return "Correct" if prop == truth else "Wrong"
        except:
            return row["proposed_answer"]
    elif row["dataset_source"] == "UMWP":
        if row["proposed_answer"] == "UNANSWERABLE":
            return "Correctly identified unanswerable" if row["ground_truth_answer"] == "UNANSWERABLE" else "Falsely claiming to be unanswerable"
        elif row["proposed_answer"] == "UNCERTAIN_SOLUTION":
            return "Rejected due to uncertainty"
        else:
            try:
                prop = float(row["proposed_answer"])
                truth = float(row["ground_truth_answer"])
                return "Correct" if prop == truth else "Wrong"
            except:
                return row["proposed_answer"]
    else:
        return "Unknown"

# Apply classification functions
df.loc[df["model"] == "NAKED_GPT", "result"] = df[df["model"] == "NAKED_GPT"].apply(classify_naked, axis=1)
df.loc[df["model"] == "NEEDLE", "result"] = df[df["model"] == "NEEDLE"].apply(classify_needle, axis=1)

# Set up the subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
models = ["NAKED_GPT", "NEEDLE"]
datasets = ["GSM8K", "CIAR", "UMWP"]

# Loop through each model and dataset
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        subset = df[(df["model"] == model) & (df["dataset_source"] == dataset)]
        counts = subset["result"].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"{model} - {dataset}")

plt.tight_layout()
plt.show()