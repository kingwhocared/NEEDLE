from pydantic import BaseModel

from utils.MyOpenAIUtils import OPENAI_CLIENT, GPT_MODEL
from utils.logging_utils import MyLoggerForFailures


class _ProblemIsAnswerableInspectionResult(BaseModel):
    answerable: bool
    reason: str


class InputCheckingAgent:
    def determine_solvable(self, solve_request, logger: MyLoggerForFailures):
        messages = [
            {"role": "system",
             "content": "Your task is to evaluate whether a given question is answerable based on logical completeness, clarity, and realism. "
                        "A question is answerable if and only if, "
                        "based only on the provided input and without any assumptions, "
                        "there is only one objectively correct numeric answer. "
                        "Analyze the question carefully. "
                        "A question is unanswerable if any of the following issues are present: "
                        "1). Key Information Missing – Essential details required to determine the answer are not provided. "
                        "2). Ambiguous Key Information – Critical elements of the question are unclear, leading to multiple possible interpretations. "
                        "3). Unrealistic Conditions – The question assumes impossible or highly improbable scenarios. "
                        "4). Unrelated Object – The question introduces an object or concept that was not previously mentioned and lacks necessary context. "
                        "5). Question Missing – The text does not contain a well-formed question."
                        "6). Open to interpretation - The problem can yield different answers based on varying perspectives. "},
            {"role": "user", "content": f"Think and inspect if the following problem is answerable. "
                                        f"Are there any signs it might not be answerable?"
                                        f": {solve_request}"}
        ]
        logger.log(f"Starting verification of input problem: {solve_request}")

        completion = OPENAI_CLIENT.beta.chat.completions.parse(
            model=GPT_MODEL,
            messages=messages,
        )

        thought_message = completion.choices[0].message
        logger.log(f"Thought about answer-ability: {thought_message.content}")
        messages.append({
            "role": "assistant",
            "content": thought_message.content
        })

        messages.append({
            "role": "user",
            "content": "Reflecting on your thoughts, is the problem answerable?"
        })

        completion = OPENAI_CLIENT.beta.chat.completions.parse(
            model=GPT_MODEL,
            messages=messages,
            response_format=_ProblemIsAnswerableInspectionResult,
        )

        inspection_result = completion.choices[0].message.parsed
        inspection_result: _ProblemIsAnswerableInspectionResult

        answerable = inspection_result.answerable
        reason = inspection_result.reason

        logger.log(f"Input inspection result: answerable={answerable}, reason={reason}")

        return answerable
