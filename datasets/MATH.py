'''
Methods to deal with the MATH dataset
'''

import re
import pandas as pd

def extract_answer(text: str) -> int | None:
    # regular expression to match the number within \\boxed{}
    pattern = r'\\boxed\{(\d+)\}'
    
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    else:
        #print("answer in boxed not found")
        return None
    
def accuracy(model_output, actual, n_samples):
    '''
    
    '''
    

