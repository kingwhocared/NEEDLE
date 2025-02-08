from pydantic import BaseModel

from utils.MyOpenAIUtils import get_openai_inference_with_schema


class _PythonEvaluableExpression(BaseModel):
    expression: str


class CalculatorAgent:

    def __init__(self):
        pass

    def serve_compute_request(self, compute_request):
        prompt_with_request = f'Return only raw text of a mathematical expression that can be evaluated in python for the following: "{compute_request}"'
        python_formatted_format = get_openai_inference_with_schema(prompt_with_request, _PythonEvaluableExpression)
        python_formatted_format = python_formatted_format.expression
        try:
            res = eval(python_formatted_format)
        except Exception as e:
            print(f"Failed to evaluate expression: {python_formatted_format}")
            raise
        return res


