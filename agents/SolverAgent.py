from utils.MyOpenAIUtils import OPENAI_CLIENT
from CalculatorAgent import CalculatorAgent
from utils.logging_utils import MyLoggerForFailures

_CALL_CALCULATOR_FUNCTION_NAME = "simple_calculator"
_CALL_ANSWER_READY_FUNCTION_NAME = "final_answer"

_LIMIT_LLM_CALLS_FOR_SOLVER_AGENT = 20


class SolverAgent:
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": _CALL_CALCULATOR_FUNCTION_NAME,
                    "description": "Compute a simple arithmetic expression and return the resulting number. "
                                   "This calculator only supports basic arithmetic: "
                                   "addition, subtraction, multiplication, and division. "
                                   "Please provide an expression that uses only numerical values "
                                   "and these four basic operations."
                                   "It is preferred you do not use more than a single operation.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": [
                            "arithmetic_task",
                            "purpose_or_meaning",
                        ],
                        "properties": {
                            "arithmetic_task": {
                                "type": "string",
                                "description": "A string representing a valid arithmetic expression using only "
                                               "addition, subtraction, multiplication, or division "
                                               "(e.g., 'how much is 8 times 9?' or 'divide 12 by 4')."
                                               "No variables, symbols, or literals allowed as input."
                                               "Do not forget about order of operations, use brackets as required."
                            },
                            "purpose_or_meaning": {
                                "type": "string",
                                "description": "A string describing the purpose or meaning of the expression."
                                               "For example, for an arithmetic_task of '40 times 3', the meaning"
                                               " could be 'the number of hours in 3 40-hour work weeks.'"
                            },
                        },
                        "additionalProperties": False
                    }
                }
            },
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
                        f"please use the {_CALL_CALCULATOR_FUNCTION_NAME} tool for that."},
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

            if tool_requested == _CALL_CALCULATOR_FUNCTION_NAME:
                compute_request = tool_arguments["arithmetic_task"]
                purpose_or_meaning = tool_arguments["purpose_or_meaning"]
                logger.log(f"LLM requested compute request: {compute_request}, 'purpose_or_meaning': {purpose_or_meaning}")
                try:
                    compute_result = self.calculator_agent.serve_compute_request(compute_request)
                except Exception as e:
                    error_message = f"Computer agent failed: {e}"
                    logger.log(error_message)
                    raise RuntimeError(error_message)
                logger.log(f"Computer returned: {compute_result}")

                messages.append(completion.message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(compute_result)
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
