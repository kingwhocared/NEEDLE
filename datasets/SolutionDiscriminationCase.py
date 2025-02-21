from dataclasses import dataclass

@dataclass
class SolutionCase:
    question: str
    answer: int
    proposed_answer: int
    proposed_solution_trace: object


