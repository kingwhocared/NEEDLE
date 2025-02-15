import unittest
from tqdm import tqdm
from pydantic import BaseModel

from datasets.GSM8K import GSM8K

from utils.logging_utils import MyLoggerForFailures
from utils.MyOpenAIUtils import get_openai_inference_with_schema, get_openai_inference

class _NumberSchema(BaseModel):
    answer: int

class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.dataset_GSM8K = GSM8K()

    def test_solver_agent_on_GSM8K(self):
        n_tests = 0
        n_successes = 0
        logger = MyLoggerForFailures(f"test_raw_llm_on_GSM8K")
        for _ in tqdm(range(100), desc="Processing"):
            try:
                q, a = self.dataset_GSM8K.get_next_GSM_question()
                n_tests += 1
                starting_test_message = f'Testing a request: \nq: {q}, a: {a}'
                logger.log(starting_test_message)
                try:
                    # ret = get_openai_inference_with_schema(q, _NumberSchema).answer
                    ret = get_openai_inference(q)
                    ret = get_openai_inference_with_schema(f"get the final answer from here as a number: {ret}", _NumberSchema).answer
                    message = f'LLM returned: {ret}'
                    logger.log(message)
                except Exception as e:
                    error_message = f"LLM returned error: {e}"
                    logger.log(error_message)
                    ret = None

                is_correct = (ret == a)
                logger.log(
                    "Answer was correct!" if is_correct else "Failure to output answer" if ret is None else "Wrong answer!")
                was_successful = is_correct

                if was_successful:
                    n_successes += 1
            except Exception as e:
                pass
        accuracy = 100 * n_successes / n_tests
        logger.log(f"For GSKM8K, being correct on {n_successes} out of {n_tests}, naked llm has accuracy {accuracy}%")
        logger.flush_log_to_file()
        self.assertLess(50, accuracy)


if __name__ == '__main__':
    unittest.main()
