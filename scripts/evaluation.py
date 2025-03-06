from tqdm import tqdm

from utils.MyOpenAIUtils import GPT_MODEL
from utils.logging_utils import MyLoggerForFailures
from utils.experiment_archiving_utils import ExperimentSample, ExperimentsArchivingUtil, INPUT_IS_UNANSWERABLE, \
    already_exists_archived_experiment_sample
from agents.NakedGptAsSolver import NakedGptAsSolver
from datasets.GSM8K import GSM8K
from datasets.UWMP import UMWP

# models
_NAKED_LLM = "NAKED_GPT"

# datasets
_GSM8K = "GSM8K"
_UMWP = "UMWP"


def run_and_archive_evaluation(experiment_name,
                               model,
                               model_version_label,
                               dataset_source,
                               n_samples,
                               ):
    experimentsArchivingUtil = ExperimentsArchivingUtil(experiment_name=experiment_name)

    if model == _NAKED_LLM:
        get_model_answer = NakedGptAsSolver.query_nakedGPT
    else:
        raise NotImplementedError(f"Invalid model {model}")

    if dataset_source == _GSM8K:
        gsm8k = GSM8K()
        n_samples = min(n_samples, gsm8k.len_dataset)

        def get_next_from_dataset_source():
            q, a, id = gsm8k.get_next_GSM_question()
            return id, q, a
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

        if already_exists_archived_experiment_sample(experiment_name, model, model_version_label, dataset_source, question_id):
            print(f"Skipping already archived question: {question}")
            continue

        logger = MyLoggerForFailures(f"q_id_{question_id}")

        try:
            proposed_answer = get_model_answer(question, logger)
        except Exception as e:
            print(f"Failed to get model answer, q:{question_id} exception:{e}")
            continue  # to next sample...

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


run_and_archive_evaluation(experiment_name="test",
                           model=_NAKED_LLM,
                           model_version_label=GPT_MODEL,
                           dataset_source=_UMWP,
                           n_samples=2,
                           )
