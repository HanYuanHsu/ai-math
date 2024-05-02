'''
Methods to deal with the MATH dataset
'''

import re
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from utils.file import load_json
import random

class MATHDataset(Dataset):
    def __init__(self, basedir, dataset_type='train', sample_per_categorie_num=10, problem_types=None):
        self.data = []

        for problem_type in os.listdir(f'{basedir}/{dataset_type}/'):
            if ( problem_types == None ) or ( problem_type in problem_types ):
                for problem_json in os.listdir(f'{basedir}/{dataset_type}/{problem_type}'):
                    problem = load_json(f'{basedir}/{dataset_type}/{problem_type}/{problem_json}')
                    answer = self.extract_answer(problem['solution'])

                    if isinstance(answer, int):
                        self.data.append({
                            'id': str(int(problem_json.replace('.json', ''))),
                            'problem': str(problem['problem']),
                            'level': str(problem['level'].replace('Level ', '')),
                            'type': str(problem['type']),
                            'answer': int(answer),
                        })
        random.shuffle(self.data)
        self.sample_per_categories(sample_per_categorie_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def extract_answer(self, text: str) -> int | None:
        # regular expression to match the number within \\boxed{}
        pattern = r'\\boxed\{(\d+)\}'
        
        match = re.search(pattern, text)
        
        if match:
            return int(match.group(1))
        else:
            return None

    def sample_per_categories(self, num):
        new_data = []
        counter = {}
        for i in self.data:
            counter[i['type']] = 1 if i['type'] not in counter else counter[i['type']] + 1
            if counter[i['type']] <= num:
                new_data.append(i)
        self.data = new_data

    def get_data(self):
        return self.data
    
    
def accuracy(model_output, actual, n_samples):
    '''
    
    '''
    

