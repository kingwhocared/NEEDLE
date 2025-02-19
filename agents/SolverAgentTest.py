import unittest
from tqdm import tqdm

from datasets.SyntheticArithmetics import SyntheticArithmetics
from datasets.GSM8K import GSM8K

from utils.logging_utils import MyLoggerForFailures
from agents.SolverAgentV2 import SolverAgent


class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.solverAgent = SolverAgent()
        self.syn_arithmetics_dataset = SyntheticArithmetics()
        self.dataset_GSM8K = GSM8K()

    def _test_a_problem_solved_by_solver_agent(self, q, a, logger):
        starting_test_message = f'Testing a solver agent request: \nq: {q}, a: {a}'
        logger.log(starting_test_message)
        try:
            ret = self.solverAgent.serve_solve_request(q, logger=logger)
            print(f'SolverAgent returned: {ret}')
        except Exception as e:
            error_message = f"Solver agent serve_solve_request threw an exception!: {e}"
            print(error_message)
            logger.log(error_message)
            ret = None

        is_correct = (ret == a)
        logger.log(
            "Answer was correct!" if is_correct else "Failure to output answer" if ret is None else "Wrong answer!")
        return is_correct

    def test_solver_agent_is_reliable_on_synthetic_dataset(self):
        n_times_tested_on_synthetic_questions = 0
        logger = MyLoggerForFailures("synthetic_dataset")
        logger.log("Starting test_solver_agent_is_reliable_on_synthetic_dataset")
        try:
            for i in range(100):
                q, a = self.syn_arithmetics_dataset.gen_arithmetics_question()
                self.assertTrue(self._test_a_problem_solved_by_solver_agent(q, a))
                n_times_tested_on_synthetic_questions += 1
        finally:
            logger.flush_log_to_file(f"Number of synthetic questions tested successfully: {n_times_tested_on_synthetic_questions}")
            logger.flush_log_to_file()

    def test_solver_agent_on_GSM8K(self):
        n_tests = 0
        n_successes = 0
        logger = MyLoggerForFailures(f"test_solver_agent_on_GSM8K")
        for _ in tqdm(range(150), desc="Processing"):
            try:
                q, a = self.dataset_GSM8K.get_next_GSM_question()
                n_tests += 1
                was_successful = self._test_a_problem_solved_by_solver_agent(q, a, logger=logger)
                if was_successful:
                    n_successes += 1
                accuracy = 100 * n_successes / n_tests
                print(f"Accuracy: {accuracy}")
            except Exception as e:
                pass
        logger.log(f"For GSKM8K, being correct on {n_successes} out of {n_tests}, solver agent has accuracy of {accuracy}%")
        logger.flush_log_to_file()
        self.assertLess(50, accuracy)


if __name__ == '__main__':
    unittest.main()
