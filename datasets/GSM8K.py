import os
import json
import random

from utils.globals import _RANDOM_SEED


class GSM8K:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "resources", "GSM8k_test.jsonl")

        with open(file_path) as fh:
            self.dataset = [json.loads(line) for line in fh.readlines() if line]
        for i, d in enumerate(self.dataset):
            d['answer'] = d['answer'].split("#### ")[1]
            d['id'] = i
        self.len_dataset = len(self.dataset)
        self._cursor = 0
        random.seed(_RANDOM_SEED)
        random.shuffle(self.dataset)

    def get_next_GSM_question(self):
        d = self.dataset[self._cursor]
        self._cursor += 1
        self._cursor %= self.len_dataset

        try:
            return d['question'], int(d['answer'].replace(",", "")), d['id']
        except Exception as e:
            print(f"Exception at getting next GSM questions: {e}")
            raise

    def get_question_with_prefix(self, prefix):
        for q in self.dataset:
            if q['question'].startswith(prefix):
                return q['question'], int(q['answer']), q['id']
        raise RuntimeError("No such question found!")
