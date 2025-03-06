from utils.experiment_archiving_utils import INPUT_IS_UNANSWERABLE, PROPOSED_OUTPUT_IS_UNCERTAIN

from agents.InputCheckingAgent import InputCheckingAgent
from agents.SolverAgent import SolverAgent
from agents.Judge import Judge


class NEEDLE:
    def __init__(self):
        self.inputChecker = InputCheckingAgent()
        self.solver = SolverAgent()
        self.judge = Judge()

    def answer_query(self, query, logger):
        answerable = self.inputChecker.determine_solvable(query, logger)

        if not answerable:
            return INPUT_IS_UNANSWERABLE
        else:
            proposed_answer, solution_trace = self.solver.serve_solve_request(query, logger)
            judge_deems_solution_okay = self.judge.verify_a_solution_trace(solution_trace, logger)
            if judge_deems_solution_okay:
                return proposed_answer
            else:
                return PROPOSED_OUTPUT_IS_UNCERTAIN
