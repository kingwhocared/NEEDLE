import os
import json
import random

from utils.globals import _RANDOM_SEED


_QUESTIONS_TO_IGNORE = [
    "In a 600 meters circular runway, Alice and Bob run from the same location clockwise. They meet with each other every 12 minitues. If they do not change the speed and run from the same location again while this time Alice runs counterclockwise, then they meet with each other every 4 minites. How long does Alice need to run one circle?",
    # ignoring because this question has 2 different possible answers.
    "In a country, all families want a boy. They keep having babies till a boy is born. What is the expected ratio of boys and girls in the country?",
    # ignoring because answer is a ratio, which is not a single definitive number.

]

class CIAR:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "resources", "CIAR.json")

        with open(file_path) as fh:
            dataset_raw = json.load(fh)
        self.dataset = []
        for i, d in enumerate(dataset_raw):
            q = d['question']
            if q in _QUESTIONS_TO_IGNORE:
                continue
            answer = None
            for ans in d['answer']:
                try:
                    answer = float(ans)
                    break
                except ValueError as e:
                    continue
            assert answer is not None
            d_formatted = {
                'id': i + 1,
                'question': q,
                'answer': answer,
            }
            self.dataset.append(d_formatted)
        self.len_dataset = len(self.dataset)
        self._cursor = 0
        random.seed(_RANDOM_SEED)
        random.shuffle(self.dataset)

    def get_next_CIAR_question(self):
        d = self.dataset[self._cursor]
        self._cursor += 1
        self._cursor %= self.len_dataset

        try:
            id = d['id']
            question = d['question']
            answer = d['answer']
            return id, question, answer
        except Exception as e:
            print(f"Exception at getting next CIAR question: {e}")
            raise

    def get_question_with_prefix(self, prefix):
        for q in self.dataset:
            if q['question'].startswith(prefix):
                return q['question'], int(q['answer']), q['id']
        raise RuntimeError("No such question found!")
