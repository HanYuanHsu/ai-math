from models.base import Model
import openai
from openai import OpenAI
import pandas as pd
import os
from datasets import MATH

import re
from sympy import symbols, simplify, Eq, solve
from latex2sympy2 import latex2sympy

default_system_prompt = "You are an assistant that is very good at mathematics. Given a mathematics problem, determine the answer, which is an integer, a float, or a fraction. Put your answer in the box \\boxed{}"
class GPT3(Model):
    def __init__(self, system_prompt=default_system_prompt):
        self.sysprompt = system_prompt

        self.client = OpenAI()
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is not None:
            openai.api_key = api_key
        else:
            raise Exception("openai api key not set")

    def _predict(self, question: str):
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.sysprompt},
                    {"role": "user", "content": "What is $(\\frac{7}{8})^3 \\cdot (\\frac{7}{8})^{-3}$?"},
                    {"role": "assistant", "content": "$\\boxed{1}$."},
                    {"role": "user", "content": "In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?"},
                    {"role": "assistant", "content": "$\\boxed{15}$."},
                    {"role": "user", "content": "Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$"},
                    {"role": "assistant", "content": "$\\boxed{\\sqrt{59}}$."},
                    {"role": "user", "content": "The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?"},
                    {"role": "assistant", "content": "$\\boxed{\\frac{1}{32}}$."},
                    {"role": "user", "content": "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?"},
                    {"role": "assistant", "content": "$\\boxed{181}$."},
                    {"role": "user", "content": "Calculate $6 \\cdot 8\\frac{1}{3}$."},
                    {"role": "assistant", "content": "$\\boxed{50}$."},
                    {"role": "user", "content": "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"},
                    {"role": "assistant", "content": "$\\boxed{2}$."},
                    {"role": "user", "content": "How many zeros are at the end of the product 25 $\\times$ 240?"},
                    {"role": "assistant", "content": "$\\boxed{3}$."},
                    {"role": "user", "content": question}
                ]
            )
        return response.choices[0].message.content


    def predict(self, test_df: pd.DataFrame, n_samples=-1) -> pd.DataFrame:        
        if n_samples >= 1:
            df = test_df.sample(n=n_samples)            
        else:
            df = test_df

        predictions = df['problem'].apply(self._predict)
        output = predictions.apply(MATH.extract_answer).apply(mod1000)

        result = pd.concat([df[['id', 'problem']], predictions, output], axis=1)
        result.columns = ['id', 'problem', 'output_lengthy', 'output']

        return result

def mod1000(output):
    if output is not None:
        return int(output) % 1000


def get_equations(text: str):
    expr_pattern = r'\$([^$]+)\$'
    eqn_pattern = r'^(.*?)\s*=\s*([^=]+).*$'
    # we will match each expression with this eqn_pattern
    # if the expression turns out to be chained equailties (more than 2 terms) 
    # then we only extract the first two terms

    expression_matches = re.finditer(expr_pattern, text) # all math expressions. The text is scanned from left to right,
    # so the expressions that appear will also follow that order
    expressions = [] # stores [expr, start_pos, end_pos]

    for match in expression_matches:
        expr = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        expressions.append([expr, start_pos, end_pos])

    equations = []
    for l in expressions:
        expr = l[0]
        match = re.match(eqn_pattern, expr)
        eqn = None
        if match:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            eqn = [lhs, rhs]
        equations.append(eqn)

    # compare lhs and rhs

    equalities = {} # a dictionary whose key is the index of the equation in `equations` array
                    # and value is boolean
    solutions = {} # a dictionary whose key is the index of the equation in `equations` array
                   # and value is a list of solutions for that equation
    parse_exceptions = set() # stores indices of the equations that fail parsing

    for i, eqn in enumerate(equations):
        if eqn is None:
            continue

        lhs = latex2sympy(eqn[0])
        rhs = latex2sympy(eqn[1])
        
        # Get variables in the expressions
        lhs_vars = lhs.free_symbols
        rhs_vars = rhs.free_symbols
        
        if lhs_vars == rhs_vars:
            # If variables are the same, check for equality
            try:
                equality_check = Eq(lhs, rhs)
                result = equality_check.simplify()
                equalities[i] = result
            except:
                parse_exceptions.add(i)
        else:
            # If variables are different, solve the equation
            try:
                solution = solve(Eq(lhs, rhs))
                solutions[i] = solution
            except:
                parse_exceptions.add(i)

    return expressions, equations, equalities, solutions, parse_exceptions


def get_hint(expressions, solutions) -> str:
    '''
    After identifying the expressions, equations, solutions, etc from get_equations,
    use them to form hints for the GPT to solve the problem faster.
    The arguments of this method comes from the outputs of get_equations method.
    '''

    hint_title = "You MUST consider the following facts, which will help you solve the question a lot:\n"
    hint = ""
    for idx, sol_list in solutions.items():
        # this contest only has integer solutions
        # so filter out non-integer solutions, like complex numbers
        sol_cleaned = []
        for sol in sol_list:
            try:
                sol_int = int(sol)
                sol_cleaned.append(str(sol_int))
            except:
                pass
        
        s = '' if len(sol_cleaned) == 1 else 's'

        hint += f"The equation {expressions[idx][0]} has solution{s} {', '.join(sol_cleaned)}\n"

    if len(hint) > 0:
        hint = hint_title + hint

    return hint
