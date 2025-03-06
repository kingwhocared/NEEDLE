import os
import json
import random

from utils.experiment_archiving_utils import INPUT_IS_UNANSWERABLE
from utils.globals import _RANDOM_SEED

class UMWP:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "resources", "UMWP.jsonl")

        with open(file_path) as fh:
            dataset_raw = [json.loads(line) for line in fh.readlines() if line]
        self.dataset = []
        for d in dataset_raw:
            answerable = d['answerable']
            answer = d['answer'][0] if answerable else INPUT_IS_UNANSWERABLE
            if answerable and not answer.is_integer():
                continue  # For now, ignoring floats. There are only 3 float cases among 5,200 questions.
            d_formatted = {
                'id': d['id'],
                'question': d['question'],
                'answer': answer,
                'answerable': answerable,
            }
            self.dataset.append(d_formatted)
        self.len_dataset = len(self.dataset)
        self._cursor = 0
        random.seed(_RANDOM_SEED)
        random.shuffle(self.dataset)

    def get_next_UWMP_question(self):
        d = self.dataset[self._cursor]
        self._cursor += 1
        self._cursor %= self.len_dataset

        try:
            id = d['id']
            question = d['question']
            answerable = d['answerable']
            answer = d['answer']
            return id, question, answerable, answer
        except Exception as e:
            print(f"Exception at getting next UWMP question: {e}")
            raise

    def get_question_with_prefix(self, prefix):
        for q in self.dataset:
            if q['question'].startswith(prefix):
                return q['question'], int(q['answer']), q['id']
        raise RuntimeError("No such question found!")
