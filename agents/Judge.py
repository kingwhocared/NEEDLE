import copy

from utils.MyOpenAIUtils import OPENAI_CLIENT, _GPT_MODEL, get_openai_inference_with_schema

_LIMIT_LLM_CALLS_FOR_INVESTIGATION = 5
_CONCLUSION_REACHED_FUNC_NAME = "reached_conclusion"


class Judge:
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": _CONCLUSION_REACHED_FUNC_NAME,
                    "description": "Called when the investigation can conclude "
                                   "the assurance of the solution final answer. "
                                   "Do not disqualify an entire solution if "
                                   "the final answer is still probably correct.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": [
                            "found_error_that_led_to_wrong_final_answer",
                            "justification",
                        ],
                        "properties": {
                            "found_error_that_led_to_wrong_final_answer": {
                                "type": "boolean",
                                "description": "The final answer numeric output is probably wrong"
                            },
                            "justification": {
                                "type": "string",
                                "description": "If there is a detection of an error, shortly describe why."
                            },
                        },
                        "additionalProperties": False
                    }
                }
            }
        ]

    def verify_a_solution_trace(self, solution_trace, logger):
        # give an estimation on the dependability of the solution.
        # True if the solutions is dependable, False otherwise

        q_to_solve = solution_trace[1]["content"].split("Please solve: ")[1]

        messages_for_interrogator = []
        messages_for_solver = copy.copy(solution_trace)

        messages_for_interrogator.append(
            {"role": "system",
             "content": "Your task is to help verify that the solution generated a correct final answer. "
                        "The solution you are given is most probably correct. "
                        "Dont be pedantic in your investigation, "
                        "If the final answer is correct, that is okay. "
                        "We only care about major fatal logical errors"
                        " about the solution misunderstanding the problem it is solving "
                        "that leads to a wrong final answer. "
                        "The solution you are given used a calculator to perform arithmetics, "
                        "and you can assume that the calculator works perfectly. "
                        "Therefore, you are looking for a fatal logical error. "
                        f"The problem was: {q_to_solve}. "
                        f"And the solution is as follows."
             }
        )

        steps_taken_by_solver = solution_trace[2:-2]
        assert len(steps_taken_by_solver) % 3 == 0  # thought, tool call, tool result
        for i in range(0, len(steps_taken_by_solver), 3):
            thought, tool_call, tool_ret = steps_taken_by_solver[i:i + 3]

            tool_arguments = eval(tool_call.tool_calls[0].function.arguments)

            operation = tool_arguments["operation"]
            num1 = tool_arguments["num1"]["numeric value"]
            num2 = tool_arguments["num2"]["numeric value"]
            calculator_return = tool_ret["content"]

            step_described_for_solver = {
                "role": "user",
                "content": f"{thought.content}\n"
                           f"Using the calculator, {num1} {operation} {num2} = {calculator_return}. "
            }
            messages_for_interrogator.append(step_described_for_solver)

        final_answer = eval(solution_trace[-1].tool_calls[0].function.arguments)['output_numerical_value']

        final_answer_message = {
            "role": "user",
            "content": f"{solution_trace[-2].content}. "
                       f"The final answer is: {final_answer}"
        }

        messages_for_interrogator.append(final_answer_message)
        messages_for_solver = messages_for_solver[:-2]
        messages_for_solver.append(final_answer_message)



        # logger.log(f"Solution trace:")
        # for m in messages_for_interrogator:
        #     logger.log(m["content"])

        n_times_prompted_investigation = 0

        # ask the investigator
        messages_for_interrogator.append({
            "role": "user",
            "content": "Generate a question to the solver to help understand "
                       "if there was an error in the solution."
        })

        completion = OPENAI_CLIENT.chat.completions.create(
            model=_GPT_MODEL,
            messages=messages_for_interrogator,
        )

        completion = completion.choices[0]
        q_to_solver = completion.message.content
        messages_for_interrogator.append(completion.message)
        logger.log(f"Question to solver: {q_to_solver}")

        messages_for_solver.append({
            "role": "user",
            "content": q_to_solver,
        })

        completion = OPENAI_CLIENT.chat.completions.create(
            model=_GPT_MODEL,
            messages=messages_for_solver,
        )

        completion = completion.choices[0]
        a_from_solver = completion.message.content
        a_from_solver_message = {
            "content": a_from_solver,
            "role": "user"
        }
        logger.log(f"Answer from solver: {a_from_solver}")

        messages_for_interrogator.append(a_from_solver_message)

        completion = OPENAI_CLIENT.chat.completions.create(
            model=_GPT_MODEL,
            messages=messages_for_interrogator,
            tools=self.tools,
            tool_choice="required"
        )

        tool_call = completion.choices[0].message.tool_calls[0]
        tool_requested = tool_call.function.name
        params = tool_call.function.arguments
        params = params.replace("true", "True").replace("false", "False")
        tool_arguments = eval(params)

        if tool_requested == _CONCLUSION_REACHED_FUNC_NAME:
            suspected_verdict = tool_arguments['found_error_that_led_to_wrong_final_answer']
            justification = tool_arguments['justification']
            if suspected_verdict:
                logger.log(f"Reached conclusion that there is a suspicion for an error: {justification}")
            else:
                logger.log(f"Reached conclusion that solution is okay: {justification}")
            return not suspected_verdict
        else:
            error_message = f"Invalid tool requested: {tool_requested}"
            logger.log(error_message)
            raise RuntimeError(error_message)

        error_message = (f"Investigator agent was called {n_times_prompted_investigation} "
                         f"times but has not able to conclude investigation.")
        logger.log(error_message)
        raise RuntimeError(error_message)
