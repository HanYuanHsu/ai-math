DEEPSEEK_PROMPT_TEMPLATE = '''
___input___
Please reason step by step, and put your final answer within \boxed{}.
'''

CODE = """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.

Approach:"""

CODE1 = """Below is a math problem you are to solve:
{}
Generate sympy code to solve the problem.
Remember:
-- Each line in the code should be clearly commented to explain why you take this step to solve the problem.
-- Write the entire script covering all the steps and wrap it within ```python ... ```.
-- The entire script should clearly demonstrate your step-by-step reasoning.
-- Print out the final result, and there must be only one print statement in the script.
-- The final result printed out must be a nonnegative integer, not an algebraic expression.

Approach:"""


COT = """Below is a math problem you are to solve (positive numerical answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""
