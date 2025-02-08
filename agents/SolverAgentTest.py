import unittest

from datasets.SyntheticArithmetics import SyntheticArithmetics
from datasets.GSM8K import GSM8K

from Utils.logging_utils import MyLoggerForFailures
from agents.SolverAgent import SolverAgent

class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.solverAgent = SolverAgent()
        self.syn_arithmetics_dataset = SyntheticArithmetics()
        self.dataset_GSM8K = GSM8K()

    def _test_a_problem_solved_by_solver_agent(self, q, a):
        print(f'Testing a solver agent request: \nq: {q}, a: {a}')
        ret = self.solverAgent.serve_solve_request(q, logger=MyLoggerForFailures(q))
        print(f'SolverAgent returned: {ret}')
        return ret == a

    def test_solver_agent_is_reliable_on_synthetic_dataset(self):
        n_times_tested_on_synthetic_questions = 0
        try:
            for i in range(100):
                q, a = self.syn_arithmetics_dataset.gen_arithmetics_question()
                self.assertTrue(self._test_a_problem_solved_by_solver_agent(q, a))
                n_times_tested_on_synthetic_questions += 1
        finally:
            print(f"Number of synthetic questions tested successfully: {n_times_tested_on_synthetic_questions}")

    def test_solver_agent_on_GSM8K(self):
        n_tests = 0
        n_successes = 0
        for i in range(1):
            try:
                q, a = self.dataset_GSM8K.get_next_GSM_question()
                n_tests += 1
                was_successful = self._test_a_problem_solved_by_solver_agent(q, a)
                if was_successful:
                    n_successes += 1
            except Exception as e:
                pass
        accuracy = 100 * n_successes / n_tests
        print(f"For GSKM8K, being correct on {n_successes} out of {n_tests}, solver agent has accuracy of {accuracy}%")
        assert accuracy > 50




if __name__ == '__main__':
    unittest.main()
