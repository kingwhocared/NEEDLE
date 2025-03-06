import unittest
from tqdm import tqdm
from sklearn.metrics import f1_score

from datasets.UWMP import UMWP

from utils.logging_utils import MyLoggerForFailures
from agents.InputCheckingAgent import InputCheckingAgent


class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.inputCheckingAgent = InputCheckingAgent()
        self.dataset_UMWP = UMWP()

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
                solver_thinks_answerable = self.inputCheckingAgent.determine_solvable(question, logger)
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
