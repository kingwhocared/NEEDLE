import unittest

from datasets.SyntheticArithmetics import SyntheticArithmetics
from agents.CalculatorAgent import CalculatorAgent

class CalculatorAgentTests(unittest.TestCase):
    def setUp(self):
        self.calculatorAgent = CalculatorAgent()
        self.syn_arithmetics_dataset = SyntheticArithmetics()

    def _assert_a_compute_request_calculator_agent(self, q, a):
        self.assertEqual(self.calculatorAgent.serve_compute_request(q), a)  # add assertion here

    def test_calculator_agent_is_reliable(self):
        n_times_tested_on_synthetic_questions = 0
        try:
            for i in range(1):
                q, a = self.syn_arithmetics_dataset.gen_arithmetics_question()
                self._assert_a_compute_request_calculator_agent(q, a)
                n_times_tested_on_synthetic_questions += 1
        finally:
            print(f"Number of synthetic questions tested: {n_times_tested_on_synthetic_questions}")



if __name__ == '__main__':
    unittest.main()
