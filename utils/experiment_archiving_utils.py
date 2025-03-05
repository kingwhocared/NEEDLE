import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import pickle
from logging_utils import MyLoggerForFailures


_PATH_TO_EXPERIMENTS = Path(__file__).parent.resolve() / "experiments"
if not os.path.exists(_PATH_TO_EXPERIMENTS):
    os.mkdir(_PATH_TO_EXPERIMENTS)

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


def _get_pickled_experiment_sample_filepath(experiment_name, experiment_sample: ExperimentSample):
    s = experiment_sample
    f_name = f"{s.model}_{s.model_version}_{s.dataset_source}_{s.question_id}.ExperimentSample"
    return os.path.join(_PATH_TO_EXPERIMENTS, experiment_name, f_name)


class ExperimentsArchivingUtil:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def serialize_and_log_experiment_end_result(self, experiment: ExperimentSample, logger: MyLoggerForFailures):
        fpath = _get_pickled_experiment_sample_filepath(self.experiment_name, experiment)
        model_returned_an_answer = experiment.proposed_answer is not None
        if model_returned_an_answer:
            with open(fpath, "wb") as f:
                pickle.dump(experiment, f)
        fpath_for_logger = fpath.replace("ExperimentSample", "_log.txt")
        logger.flush_log_to_file(filepath=fpath_for_logger)
