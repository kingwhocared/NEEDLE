import unittest
import pickle
from tqdm import tqdm

from utils.logging_utils import MyLoggerForFailures
from datasets.SolutionDiscriminationCase import SolutionCase
from agents.Judge import Judge


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = Judge()
        with open("../datasets/proposed_solutions.pkl", "rb") as f:
            self.reflection_cases = pickle.load(f)

    def test_reflection(self):
        n_tests = 0
        n_correct_solutions = 0
        n_wrong_solutions = 0
        n_accurately_reflected_correct_solution = 0
        n_accurately_reflected_wrong_solution = 0

        logger = MyLoggerForFailures(f"test_judge")
        for solution_case in tqdm(self.reflection_cases, desc="Processing"):
            print("\n")
            solution_case: SolutionCase
            try:
                solution_was_correct = (solution_case.answer == solution_case.proposed_answer)
                solution_is_thought_to_be_solid = (
                    self.judge.verify_a_solution_trace(solution_case.proposed_solution_trace, logger))

                if solution_was_correct:
                    n_correct_solutions += 1
                    if solution_is_thought_to_be_solid:
                        n_accurately_reflected_correct_solution += 1
                else:
                    n_wrong_solutions += 1
                    if not solution_is_thought_to_be_solid:
                        n_accurately_reflected_wrong_solution += 1

                if solution_was_correct != solution_is_thought_to_be_solid:
                    logger.log(f"Judge made the wrong decision.")

                n_tests += 1
                logger.log(f"Solutions tested: {n_tests}"
                           f"\n"
                           f"Number of Wrong|Right cases: {n_wrong_solutions}|{n_correct_solutions}")

                if n_wrong_solutions != 0:
                    wrong_solution_detection_accuracy = 100. * n_accurately_reflected_wrong_solution / n_wrong_solutions
                    logger.log(f"Wrong solution detection accuracy: {wrong_solution_detection_accuracy}")

                if n_correct_solutions != 0:
                    correct_solution_detection_accuracy = 100. * n_accurately_reflected_correct_solution / n_correct_solutions
                    logger.log(f"Correct solution detection accuracy: {correct_solution_detection_accuracy}")

            except ValueError as e:
                logger.log(f"Exception raised while processing case!: {e}")

        logger.flush_log_to_file()
        self.assertLess(80, wrong_solution_detection_accuracy)
        self.assertLess(80, correct_solution_detection_accuracy)


if __name__ == '__main__':
    unittest.main()
