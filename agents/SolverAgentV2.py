from pydantic import BaseModel
import numpy as np

from utils.MyOpenAIUtils import OPENAI_CLIENT
from CalculatorAgent import CalculatorAgent
from utils.logging_utils import MyLoggerForFailures

_CALL_ANSWER_READY_FUNCTION_NAME = "final_answer"

_LIMIT_LLM_CALLS_FOR_SOLVER_AGENT = 30

_GPT_MODEL = "gpt-4o-mini"

class _LogicalErrorInspection(BaseModel):
    logical_error_detected: bool
    logical_error_description: str


class SolverAgent:
    @staticmethod
    def calculator(operation, num1, num2):
        try:
            num1 = float(num1)
            num2 = float(num2)
            if operation == "add":
                return num1 + num2
            elif operation == "subtract":
                return num1 - num2
            elif operation == "multiply":
                return num1 * num2
            elif operation == "divide":
                return num1 / num2 if num2 != 0 else "Error: Division by zero"
            else:
                return "Error: Invalid operation"
        except ValueError:
            return "Error: Invalid numbers"

    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform basic arithmetic operations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meaning of operation result": {
                                "type": "string",
                                "description": "A string describing the meaning of the value of the operation."
                                               "For example, for an arithmetic operation of '40 multiply 3', the meaning"
                                               " could be 'the number of hours in 3 40-hour work weeks.'"
                            },
                            "num1": {
                                "type": "object",
                                "properties": {
                                    "num1 meaning": {
                                        "type": "string",
                                        "description": "Meaning of num1"
                                    },
                                    "numeric value": {
                                        "type": "number",
                                        "description": "Numeric value of num1"
                                    },
                                },
                                "required": ["num1 meaning", "numeric value"]
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The arithmetic operation to perform"
                            },
                            "num2": {
                                "type": "object",
                                "properties": {
                                    "num2 meaning": {
                                        "type": "string",
                                        "description": "Meaning of num2"
                                    },
                                    "numeric value": {
                                        "type": "number",
                                        "description": "Numeric value of num2"
                                    },
                                },
                                "required": ["num2 meaning", "numeric value"]
                            },
                        },
                        "required": ["meaning of operation result", "num1", "operation", "num2"]
                    }
                }
            }
            ,
            {
                "type": "function",
                "function": {
                    "name": _CALL_ANSWER_READY_FUNCTION_NAME,
                    "description": "Called when the final answer is ready "
                                   "and is being outputted via this function call.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": [
                            "output_numerical_value",
                            "output_verbal_answer",
                        ],
                        "properties": {
                            "output_numerical_value": {
                                "type": "number",
                                "description": "The final answer numeric output value"
                            },
                            "output_verbal_answer": {
                                "type": "string",
                                "description": "The final answer output verbally said."
                            },
                        },
                        "additionalProperties": False
                    }
                }
            }
        ]
        self.calculator_agent = CalculatorAgent()

    def serve_solve_request(self, solve_request, logger: MyLoggerForFailures):
        messages = [
            {"role": "system",
             "content": "Your task is to solve a question. "
                        "Each time you need to perform any arithmetics, "
                        f"please use the calculator tool for that."},
            {"role": "user", "content": f"Please solve: {solve_request}"}
        ]
        logger.log("Starting new solve request.")
        logger.log(messages)

        n_times_prompted_agent = 0

        while n_times_prompted_agent < _LIMIT_LLM_CALLS_FOR_SOLVER_AGENT:
            # prepend a thinking request for next step
            thought_request = {
                "role": "user",
                "content": "Think about your next step."
            }
            completion = OPENAI_CLIENT.chat.completions.create(
                model=_GPT_MODEL,
                messages=messages + [thought_request],
            )
            thought_message = completion.choices[0].message
            logger.log(f"LLM next thought: {thought_message.content}")
            messages.append(thought_message)

            completion = OPENAI_CLIENT.chat.completions.create(
                model=_GPT_MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice='required',
                parallel_tool_calls=False,
            )
            n_times_prompted_agent += 1

            completion = completion.choices[0]
            tool_call = completion.message.tool_calls[0]
            tool_requested = tool_call.function.name
            tool_arguments = eval(tool_call.function.arguments)

            if tool_requested == "calculator":
                messages.append(completion.message)

                operation = tool_arguments["operation"]

                try:
                    num1 = tool_arguments["num1"]["numeric value"]
                    num2 = tool_arguments["num2"]["numeric value"]
                    purpose_or_meaning = tool_arguments["meaning of operation result"]
                    num1_meaning = tool_arguments["num1"]["num1 meaning"]
                    num2_meaning = tool_arguments["num2"]["num2 meaning"]
                except KeyError as e:
                    logger.log(f"Error in calculator formatting request: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error in formatting, missing param: {e}"
                    })
                    continue

                compute_request = (operation, num1, num2)
                request_explanation = (purpose_or_meaning, num1_meaning, num2_meaning)
                logger.log(
                    f"LLM requested compute request: {compute_request}, 'request_explanation': {request_explanation}")

                try:
                    compute_result = self.calculator(*compute_request)
                    logger.log(f"Computer returned: {compute_result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(compute_result)
                    })
                except Exception as e:
                    error_message = f"calculator failed: {e}"
                    logger.log(error_message)
                    raise RuntimeError(error_message)

            elif tool_requested == _CALL_ANSWER_READY_FUNCTION_NAME:
                final_answer = tool_arguments['output_numerical_value']
                logger.log(f"Solver returned final answer: {final_answer}")
                return final_answer
            else:
                error_message = f"Invalid tool requested: {tool_requested}"
                logger.log(error_message)
                raise RuntimeError(error_message)

        error_message = f"Solver agent was called {n_times_prompted_agent} times but has not able to complete task."
        logger.log(error_message)
        raise RuntimeError(error_message)
