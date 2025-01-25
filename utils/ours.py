import re
import sys
import subprocess

def extract_python_blocks(input_string: str):
    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches]

def process_code(code: str):
    '''
    Executes a piece of code and returns its output.
    
    Args:
    code (str): code that must have a singe print statement that prints out the output of the code. 
                The value printed out will be the output of this function.

    Returns:
    The content of the print statement in the code, which is usually a string. To convert the type
    of this function's output to int, try `round(float(eval(return_value))) % 1000`.
    If any error occurs when executing the code, return -1. 
    '''
    def return_last_print(output, n):
        lines = output.strip().split('\n')
        if lines:
            return lines[n]
        else:
            return ""

    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
        
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    
    with open('_tmp.py', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 7 ' + sys.executable + ' _tmp.py'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        return return_value
    except Exception as e:
        print(f"error: {e}")
        return -1

if __name__ == "__main__":
    code1 = '''
from sympy import factorial

def seating_arrangements():
    # Total number of ways to arrange the 7 children without any restrictions
    total = factorial(7)

    # Number of ways to arrange the 7 children such that no two boys are next to each other
    no_boys_together = factorial(4) / factorial(2) * factorial(3)

    # Number of ways to arrange the 7 children such that at least two boys are next to each other
    at_least_two_boys_together = total - no_boys_together

    return at_least_two_boys_together

result = seating_arrangements()
print(result)'''
    print(f"code:{code1}")
    result = process_code(code1)
    print(f"result:{result}")

    print("----------")

    code2 = '''
print(3)
'''
    print(f"code:{code2}")
    result = process_code(code2)
    print(f"result:{result}")

    print("----------")


    code3 = '''
while True:
    pass
'''
    print(f"code:{code3}")
    result = process_code(code3)
    print(f"result:{result}")
