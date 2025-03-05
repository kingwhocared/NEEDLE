import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import pickle
from .logging_utils import MyLoggerForFailures

_PATH_TO_EXPERIMENTS = Path(__file__).parent.parent.resolve() / "experiments"
if not os.path.exists(_PATH_TO_EXPERIMENTS):
    os.mkdir(_PATH_TO_EXPERIMENTS)

COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION = "COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION"
INPUT_IS_UNANSWERABLE = "UNANSWERABLE"
PROPOSED_OUTPUT_IS_UNCERTAIN = "UNCERTAIN_SOLUTION"


@dataclass
class ExperimentSample:
    model: str
    model_version: str
    dataset_source: str
    question_id: int
    question: str
    ground_truth_answer: Union[int, str]
    proposed_answer: Union[int, str, None]


def _get_pickled_experiment_sample_filepath(experiment_name, model, model_version, dataset_source, question_id):
    f_name = f"{model}_{model_version}_{dataset_source}_{question_id}_ExperimentSample.pkl"
    return os.path.join(_PATH_TO_EXPERIMENTS, experiment_name, f_name)


def already_exists_archived_experiment_sample(experiment_name, model, model_version, dataset_source, question_id):
    fpath = _get_pickled_experiment_sample_filepath(experiment_name, model, model_version, dataset_source, question_id)
    return os.path.exists(fpath)


class ExperimentsArchivingUtil:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        path_to_experiment_folder = _PATH_TO_EXPERIMENTS / experiment_name
        if not os.path.exists(path_to_experiment_folder):
            os.mkdir(path_to_experiment_folder)

    def serialize_and_log_experiment_end_result(self, experiment: ExperimentSample, logger: MyLoggerForFailures):
        s = experiment
        fpath = _get_pickled_experiment_sample_filepath(self.experiment_name, s.model, s.model_version,
                                                        s.dataset_source, s.question_id)
        fpath_for_logger = fpath.replace("ExperimentSample.pkl", "_log.txt")
        model_returned_an_answer = experiment.proposed_answer is not None
        if model_returned_an_answer:
            with open(fpath, "wb") as f:
                pickle.dump(experiment, f)
        else:
            fpath_for_logger = fpath_for_logger.replace("_log.txt", "_failure_log.txt")
        logger.flush_log_to_file(filepath=fpath_for_logger)
