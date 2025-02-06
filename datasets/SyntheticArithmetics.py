import random

class SyntheticArithmetics:
    def __init__(self):
        pass

    def _rand_num(self):
        return random.randint(1, 1_000)

    def gen_arithmetics_question(self):
        arithmetics_problems = [
            self._addition_question,
            self._subtraction_question,
            self._multiplication_question,
            self._division_question,
        ]

        return random.choice(arithmetics_problems)()

    def _addition_question(self):
        a, b = self._rand_num(), self._rand_num()

        addition_templates = [
            "how much is {} plus {}?",
            "add {} to {}",
        ]

        return random.choice(addition_templates).format(a, b), a + b

    def _subtraction_question(self):
        a, b = self._rand_num(), self._rand_num()

        subtraction_templates = [
            "how much is {a} minus {b}?"
            "subtract {b} from {a}",
            "{a} - {b}",
        ]

        return random.choice(subtraction_templates).format(a=a, b=b), a - b


    def _multiplication_question(self):
        a, b = self._rand_num(), self._rand_num()

        multiplication_templates = [
            "how much is {} times {}?",
            "multiply {} with {}",
            "{}*{}",
        ]

        return random.choice(multiplication_templates).format(a, b), a * b

    def _division_question(self):
        c, b = self._rand_num(), self._rand_num()
        a = b * c

        division_templates = [
            "how much is {} divided by {}?",
            "divide {} by {}",
        ]

        return random.choice(division_templates).format(a, b), c
