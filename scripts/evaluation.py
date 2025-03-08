from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from dataclasses import asdict


from datasets.CIAR import CIAR
from utils.logging_utils import MyLoggerForFailures
from utils.experiment_archiving_utils import ExperimentSample, ExperimentsArchivingUtil, INPUT_IS_UNANSWERABLE, \
    already_exists_archived_experiment_sample, PATH_TO_EXPERIMENTS
from agents.NakedGptAsSolver import NakedGptAsSolver
from NEEDLE import NEEDLE
from datasets.GSM8K import GSM8K
from datasets.UWMP import UMWP

# models
_NAKED_LLM = "NAKED_GPT"
_NEEDLE = "NEEDLE"

# datasets
_GSM8K = "GSM8K"
_CIAR = "CIAR"
_UMWP = "UMWP"
_ALL_DATASETS = [_GSM8K, _CIAR, _UMWP]


def run_and_archive_evaluation(experiment_name,
                               model,
                               model_version_label,
                               dataset_source,
                               n_samples,
                               ):
    experimentsArchivingUtil = ExperimentsArchivingUtil(experiment_name=experiment_name)

    if model == _NAKED_LLM:
        get_model_answer = NakedGptAsSolver.query_nakedGPT
    elif model == _NEEDLE:
        needle = NEEDLE()
        get_model_answer = needle.answer_query
    else:
        raise NotImplementedError(f"Invalid model {model}")

    if dataset_source == _GSM8K:
        gsm8k = GSM8K()
        n_samples = min(n_samples, gsm8k.len_dataset)

        def get_next_from_dataset_source():
            q, a, id = gsm8k.get_next_GSM_question()
            return id, q, a
    elif dataset_source == _CIAR:
        ciar = CIAR()
        n_samples = min(n_samples, ciar.len_dataset)

        def get_next_from_dataset_source():
            id, question, answer = ciar.get_next_CIAR_question()
            return id, question, answer
    elif dataset_source == _UMWP:
        umwp = UMWP()
        n_samples = min(n_samples, umwp.len_dataset)

        def get_next_from_dataset_source():
            id, question, answerable, answer = umwp.get_next_UWMP_question()
            if not answerable:
                answer = INPUT_IS_UNANSWERABLE
            return id, question, answer
    else:
        raise NotImplementedError(f"Invalid datasource {dataset_source}")

    for _ in tqdm(range(n_samples), desc="Processing"):
        question_id, question, ground_truth_answer = get_next_from_dataset_source()

        if already_exists_archived_experiment_sample(experiment_name, model, model_version_label, dataset_source,
                                                     question_id):
            print(f"Skipping already archived question: {question}")
            continue

        logger = MyLoggerForFailures(f"q_id_{question_id}")

        try:
            proposed_answer = get_model_answer(question, logger)
        except Exception as e:
            print(f"Failed to get model answer, q:{question_id} exception:{e}")
            proposed_answer = None

        logger.log(f"The question was: {question}")
        logger.log(f"The ground truth answer for this question is: {ground_truth_answer}")
        logger.log(f"The answer given by the model is: {proposed_answer}")


        experimentSample = ExperimentSample(
            model=model,
            model_version=model_version_label,
            dataset_source=dataset_source,
            question_id=question_id,
            question=question,
            ground_truth_answer=ground_truth_answer,
            proposed_answer=proposed_answer
        )

        experimentsArchivingUtil.serialize_and_log_experiment_end_result(experimentSample, logger)


# for dataset in _ALL_DATASETS:
#     n_samples = 50
#     run_and_archive_evaluation(experiment_name="first_eval_NEEDLE_all_datasets",
#                                model=_NEEDLE,
#                                model_version_label="v1",
#                                dataset_source=dataset,
#                                n_samples=50,
#                                )


def get_collected_eval_from_experiments(experiments):
    all_experiments = []
    for experiment in experiments:
        folder_path = PATH_TO_EXPERIMENTS / experiment
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix == ".pkl":
                with open(file, "rb") as f:
                    sample = pickle.load(f)
                    all_experiments.append(sample)
    df = pd.DataFrame([asdict(sample) for sample in all_experiments])

    # Rounding floats so comparison can be much easier later.
    def relative_closeness(a, b):
        try:
            a_num, b_num = float(a), float(b)  # Convert to float
            return 1 - abs(a_num - b_num) / max(abs(a_num), abs(b_num))
        except ValueError:
            return np.nan  # Return NaN for non-numeric cases
    #
    df["Relative_Closeness"] = df.apply(lambda row: relative_closeness(row["proposed_answer"], row["ground_truth_answer"]), axis=1)
    to_round = df["Relative_Closeness"] >= 0.99
    df[to_round]["proposed_answer"] = df[to_round]["ground_truth_answer"]
    df.drop(columns="Relative_Closeness", inplace=True)

    return df


df = get_collected_eval_from_experiments(
    ["first_eval_NEEDLE_all_datasets",
     "first_eval_naked_all_datasets",
     ]
)

df.to_csv("df.csv", index=False)