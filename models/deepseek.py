from models.base import Model, mod1000
import pandas as pd
import requests
from datasets import MATH

class DeepSeek(Model):
    def __init__(self, host, port):
        self.url = f'http://{host}:{port}/'

    def _predict(self, question: str) -> str:
        return requests.post(self.url, data={'question': question}).json()['response']
        
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

