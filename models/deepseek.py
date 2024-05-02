from typing import Tuple
from models.base import Model, mod1000
import pandas as pd
import requests
from datasets import MATH
from datasets.MATH import extract_answer
from prompts.prompt import DEEPSEEK_PROMPT_TEMPLATE

class DeepSeek(Model):
    def __init__(self, host, port):
        self.url = f'http://{host}:{port}/'

    def _predict(self, question: str) -> str:
        prompt = DEEPSEEK_PROMPT_TEMPLATE.replace('___input___', question)
        return requests.post(self.url, data={'prompt': prompt}).json()['response']
        
    def predict(self, question: str) :
        prompt = DEEPSEEK_PROMPT_TEMPLATE.replace('___input___', question)
        prediction = self._predict(prompt)
        output = MATH.extract_answer(prediction)
        output = mod1000(output)

        return prediction, output

