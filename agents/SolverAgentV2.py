from pydantic import BaseModel
import numpy as np

from utils.MyOpenAIUtils import OPENAI_CLIENT
from CalculatorAgent import CalculatorAgent
from utils.logging_utils import MyLoggerForFailures

_CALL_ANSWER_READY_FUNCTION_NAME = "final_answer"
_SOLVE_LINEAR_EQUATION_FUNCTION_NAME = "solve_linear_equations"

_LIMIT_LLM_CALLS_FOR_SOLVER_AGENT = 15


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

    @staticmethod
    def solve_linear_equations(coefficients, constants):
        """
        Solves a system of linear equations using NumPy.

        Parameters:
        coefficients (list of lists): A 2D list representing the coefficient matrix.
        constants (list): A 1D list representing the constants.

        Returns:
        dict: A solution dictionary or an error message.
        """
        try:
            A = np.array(coefficients, dtype=float)
            B = np.array(constants, dtype=float)

            # Check if the system is solvable
            if np.linalg.det(A) == 0:
                return {"error": "The system has no unique solution (singular matrix)."}

            # Solve the system
            solution = np.linalg.solve(A, B)
            return {"solution": solution.tolist()}

        except Exception as e:
            return {"error": str(e)}

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
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The arithmetic operation to perform"
                            },
                            "num1 meaning": {
                                "type": "string",
                                "description": "Meaning of the first number"
                            },
                            "num1": {
                                "type": "number",
                                "description": "The first number"
                            },
                            "num2 meaning": {
                                "type": "string",
                                "description": "Meaning of the second number"
                            },
                            "num2": {
                                "type": "number",
                                "description": "The second number"
                            },
                        },
                        "required": ["meaning of operation result", "operation", "num1 meaning", "num1", "num2 meaning",
                                     "num2"]
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
            ,
            {
                "type": "function",
                "function": {
                    "name": _SOLVE_LINEAR_EQUATION_FUNCTION_NAME,
                    "description": "Solves a system of linear equations.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": ["coefficients", "constants"],
                        "properties": {
                            "coefficients": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "description": "Coefficient matrix of the system."
                            },
                            "constants": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Constants of the system."
                            }
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
             "content": "Your task is to solve a question."
                        # " Each time you need to perform any arithmetics, "
                        # f"you may choose to use the calculator tool for that. "
                        # f"You also have access to a linear equation tool."
             },
            {"role": "user", "content": f"Please solve: {solve_request}"}
        ]
        logger.log("Starting new solve request.")
        logger.log(messages)

        n_times_prompted_agent = 0

        while n_times_prompted_agent < _LIMIT_LLM_CALLS_FOR_SOLVER_AGENT:
            completion = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
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
                num1 = tool_arguments["num1"]
                num2 = tool_arguments["num2"]
                compute_request = (operation, num1, num2)
                purpose_or_meaning = tool_arguments["meaning of operation result"]
                num1_meaning = tool_arguments["num1 meaning"]
                num2_meaning = tool_arguments["num2 meaning"]
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
                    error_message = f"Computer agent failed: {e}"
                    logger.log(error_message)
                    raise RuntimeError(error_message)

            elif tool_requested == _SOLVE_LINEAR_EQUATION_FUNCTION_NAME:
                constants = tool_arguments["constants"]
                coefficients = tool_arguments["coefficients"]

                logger.log(
                    f"LLM requested linear eq request: {lin_eq_result}, 'request_explanation': {lin_eq_result}")
                lin_eq_result = self.solve_linear_equations(coefficients, constants)
                logger.log(f"Linear eq tool returned: {lin_eq_result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(lin_eq_result)
                })

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
