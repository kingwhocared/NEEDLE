import os
import json
import random


class GSM8K:
    def __init__(self, shuffle=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "resources", "GSM8k_test.jsonl")

        with open(file_path) as fh:
            self.dataset = [json.loads(line) for line in fh.readlines() if line]
        for d in self.dataset:
            d['answer'] = d['answer'].split("#### ")[1]
        self.len_dataset = len(self.dataset)
        self._cursor = 0
        if shuffle:
            random.shuffle(self.dataset)

    def get_next_GSM_question(self):
        d = self.dataset[self._cursor]
        self._cursor += 1
        self._cursor %= self.len_dataset

        return d['question'], int(d['answer'])
