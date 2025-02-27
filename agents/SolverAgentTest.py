import unittest
from tqdm import tqdm
from sklearn.metrics import f1_score

from datasets.SyntheticArithmetics import SyntheticArithmetics
from datasets.GSM8K import GSM8K
from datasets.UWMP import UMWP, UNANSWERABLE

from utils.logging_utils import MyLoggerForFailures
from agents.SolverAgentWithInputChecking import SolverAgentWithInputChecking


class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.solverAgent = SolverAgentWithInputChecking()
        self.syn_arithmetics_dataset = SyntheticArithmetics()
        self.dataset_GSM8K = GSM8K()
        self.dataset_UMWP = UMWP()

    def _run_a_problem_on_solver_agent(self, q, logger):
        starting_test_message = f'Testing a solver agent request: \nq: {q}'
        logger.log(starting_test_message)
        try:
            ret, _ = self.solverAgent.serve_solve_request(q, logger=logger)
            print(f'SolverAgent returned: {ret}')
        except Exception as e:
            error_message = f"Solver agent serve_solve_request threw an exception!: {e}"
            print(error_message)
            logger.log(error_message)
            ret = None

        return ret

    def _test_a_problem_solved_by_solver_agent(self, q, a, logger):
        ret = self._run_a_problem_on_solver_agent(q, logger)
        logger.log(f"Ground truth answer is: {a}")
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

    def test_solver_agent_on_GSM8K(self):
        n_tests = 0
        n_successes = 0
        logger = MyLoggerForFailures(f"test_solver_agent_on_GSM8K")
        for _ in tqdm(range(200), desc="Processing"):
            try:
                q, a = self.dataset_GSM8K.get_next_GSM_question()
                n_tests += 1
                was_successful = self._test_a_problem_solved_by_solver_agent(q, a, logger=logger)
                if was_successful:
                    n_successes += 1
                accuracy = 100 * n_successes / n_tests
                print(f"Accuracy: {accuracy}")
            except Exception as e:
                logger.log(f"Exception raised while processing GSM8K question!: {e}")
        logger.log(f"For GSKM8K, being correct on {n_successes} out of {n_tests}, solver agent has accuracy of {accuracy}%")
        logger.flush_log_to_file()
        self.assertLess(90, accuracy)

    def test_solver_agent_on_a_specified_GSM8K_question(self, q="The zookeeper feeds all the apes in the zoo. He "):
        logger = MyLoggerForFailures(q)
        q, a = self.dataset_GSM8K.get_question_with_prefix(q)
        was_successful = self._test_a_problem_solved_by_solver_agent(q, a, logger=logger)
        logger.log("Done!")


    def test_solver_agent_ability_to_detect_unanswerable(self):
        n_tests = 0
        n_successes = 0
        y_true = []  # Ground truth (actual answerable labels)
        y_pred = []  # Predicted answerable labels

        logger = MyLoggerForFailures(f"test_solver_agent_on_UMWP")
        for _ in tqdm(range(250), desc="Processing"):
            try:
                id, question, answerable, answer = self.dataset_UMWP.get_next_UWMP_question()
                n_tests += 1
                solver_thinks_answerable = self.solverAgent.determine_solvable(question, logger)
                was_successful = (solver_thinks_answerable == answerable)
                y_true.append(answerable)
                y_pred.append(solver_thinks_answerable)
                if was_successful:
                    n_successes += 1
                logger.log(f"Correct!" if was_successful else "Wrong!")
                accuracy = 100 * n_successes / n_tests
                logger.log(f"Accuracy: {accuracy:.2f}%")
                f1 = 100 * f1_score(y_true, y_pred)
                logger.log(f"F1 score: {f1:.2f}%")
            except Exception as e:
                logger.log(f"Exception raised while processing UMWP question!: {e}")
        logger.log(f"For UMWP, detecting answer-ability, f1={f1:.2f}% accuracy={accuracy:.2f}%")
        logger.flush_log_to_file()
        self.assertLess(90, accuracy)
        self.assertLess(90, f1)


if __name__ == '__main__':
    unittest.main()
