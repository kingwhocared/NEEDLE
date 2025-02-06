from Utils.MyOpenAIUtils import get_openai_inference

class CalculatorAgent:
    def __init__(self):
        pass

    def serve_compute_request(self, compute_request):
        prompt_with_request = f'Return only a raw mathematical expression text that can be evaluated in python for the following: "{compute_request}"'
        python_formatted_format = get_openai_inference(prompt_with_request)
        try:
            res = eval(python_formatted_format)
        except Exception as e:
            print(f"Failed to evaluate expression: {python_formatted_format}")
        return res


