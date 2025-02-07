import unittest

from datasets.SyntheticArithmetics import SyntheticArithmetics
from datasets.GSM8K import GSM8K

from agents.SolverAgent import SolverAgent

class SolverAgentTests(unittest.TestCase):
    def setUp(self):
        self.solverAgent = SolverAgent()
        self.syn_arithmetics_dataset = SyntheticArithmetics()
        self.dataset_GSM8K = GSM8K()

    def _assert_a_single_problem_solved_by_solver_agent(self, q, a):
        print(f'Testing a solver agent request: \nq: {q}, a: {a}')
        ret = self.solverAgent.serve_solve_request(q)
        print(f'SolverAgent returned: {ret}')
        self.assertEqual(ret, a)

    def test_solver_agent_is_reliable_on_synthetic_dataset(self):
        n_times_tested_on_synthetic_questions = 0
        try:
            for i in range(100):
                q, a = self.syn_arithmetics_dataset.gen_arithmetics_question()
                self._assert_a_single_problem_solved_by_solver_agent(q, a)
                n_times_tested_on_synthetic_questions += 1
        finally:
            print(f"Number of synthetic questions tested successfully: {n_times_tested_on_synthetic_questions}")



if __name__ == '__main__':
    unittest.main()
