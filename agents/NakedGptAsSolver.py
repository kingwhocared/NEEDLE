import numbers

from pydantic import BaseModel

from utils.logging_utils import MyLoggerForFailures
from utils.MyOpenAIUtils import get_openai_inference_with_schema, get_openai_inference
from utils.experiment_archiving_utils import COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION

class _NumberSchema(BaseModel):
    answer: float


class NakedGptAsSolver:
    @staticmethod
    def query_nakedGPT(question, logger: MyLoggerForFailures):
        try:
            starting_test_message = (f'Testing a request on raw GPT: \n'
                                     f'question: {question}')
            logger.log(starting_test_message)
            ret = get_openai_inference(question)
            logger.log(f"LLM: {ret}")
            ret = get_openai_inference_with_schema(
                f"Retrieve the final answer as an number, "
                f"or return -1 if it's missing or not a numeric value. "
                f"The context is: {ret}",
                _NumberSchema).answer
            # P.s, I looked it up, none of the answers in any of the datasets is -1 so we can use that value.
            ret = COULD_NOT_EXTRACT_NUMBER_FROM_SOLUTION if ret == -1 else ret
            message = f'LLM returned final answer: {ret}'
            logger.log(message)
            return ret
        except Exception as e:
            logger.log(f"Exception: {e}")
            return None
