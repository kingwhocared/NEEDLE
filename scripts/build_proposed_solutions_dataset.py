import pickle

from datasets.SolutionDiscriminationCase import SolutionCase
from utils.logging_utils import MyLoggerForFailures
from datasets.GSM8K import GSM8K
from agents.SolverAgent import SolverAgent

_HARD_GSM8K_QUESTIONS_MANUALLY_CHOSEN = [
    "Tim gets a promotion that offers him a 5% raise on his $20000 a month salary.",
    "Bob wants to dig a hole 6 feet long by 4 feet wide by 3 feet deep.",
    "Olivia uploaded 72 pictures to Facebook.  She put the same number of the pics into 8 albums.",
    "Nine of the kids in Gina's class are allergic to dairy, 6 are allergic to peanuts and 3 are allergic to both.",
    "Britany records 18 4-minute TikTok videos each week. She spends 2 hours a week writing amateur ",
    "Steve decides to start eating more tomatoes and decides to grows his own cherry tomatoes.",
    "Carolyn works for a delivery service company that hires on a contract basis.",
    "A custodian has to clean a school with 80 classrooms. They have 5 days to get it done.",
    "When the water is cold Ray swims a mile in 16 minutes. When the water is warm Ray swims a mile",
    "Candy has a chair rental business. During the weekdays, 60 chairs are rented each day; but during weekends,",
    "Frankie watches TV after he finishes his homework every night. On Monday and Tuesday, he watched a 1-hour",
    "Vince can staple 30 reports every 15 minutes.  If he was stapling reports from 8:00 AM until 11:00 PM,",
    "A new arcade opens up and Jack decides to play with his 3 friends.  Jack can play a game with 1",
]

_DESIRED_RATIO_OF_CORRECT_SOLUTION_CASES_TO_WRONG = 1  # for every wrong solution

_logger = MyLoggerForFailures("build_proposed_solutions_dataset")
_solver = SolverAgent()
_gsm8k = GSM8K()

cases = []

def _collect_case(q=None):
    if q is None:
        q, a, _ = _gsm8k.get_next_GSM_question()
    else:
        q, a, _ = _gsm8k.get_question_with_prefix(q)
    sol_result, sol_trace = _solver.serve_solve_request(q, _logger)
    case = SolutionCase(q, a, sol_result, sol_trace)
    cases.append(case)

for q in _HARD_GSM8K_QUESTIONS_MANUALLY_CHOSEN:
    _collect_case(q)

for _ in range(_DESIRED_RATIO_OF_CORRECT_SOLUTION_CASES_TO_WRONG * len(_HARD_GSM8K_QUESTIONS_MANUALLY_CHOSEN)):
    _collect_case()

with open("../datasets/proposed_solutions.pkl", "wb") as f:
    pickle.dump(cases, f)
