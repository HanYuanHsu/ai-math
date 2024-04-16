'''
Methods to deal with the MATH dataset
'''

import re

def extract_answer(text: str) -> int | None:
    # regular expression to match the number within \\boxed{}
    pattern = r'\\boxed\{(\d+)\}'
    
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    else:
        print("answer in boxed not found")
        return None
