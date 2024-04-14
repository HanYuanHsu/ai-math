from base import Model
from openai import OpenAI
import pandas as pd

class GPT3(Model):
    def __init__(self):
        self.client = OpenAI()

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        def predict_from_row(row):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that is very good at mathematics. Given a mathematics problem, determine the answer. Simplify your answer as much as possible."},
                    {"role": "user", "content": "What is $(\\frac{7}{8})^3 \\cdot (\\frac{7}{8})^{-3}$?"},
                    {"role": "assistant", "content": "$1$."},
                    {"role": "user", "content": "In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?"},
                    {"role": "assistant", "content": "$15$."},
                    {"role": "user", "content": "Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$"},
                    {"role": "assistant", "content": "$\\sqrt{59}$."},
                    {"role": "user", "content": "The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?"},
                    {"role": "assistant", "content": "$\\frac{1}{32}$."},
                    {"role": "user", "content": "The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?"},
                    {"role": "assistant", "content": "$181$."},
                    {"role": "user", "content": "Calculate $6 \\cdot 8\\frac{1}{3}$."},
                    {"role": "assistant", "content": "$50$."},
                    {"role": "user", "content": "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"},
                    {"role": "assistant", "content": "$2$."},
                    {"role": "user", "content": "How many zeros are at the end of the product 25 $\\times$ 240?"},
                    {"role": "assistant", "content": "$3$."},
                    {"role": "user", "content": row['problem']}
                ]
            )

            return response.choices[0].message.content
        
        predictions = test_df.apply(predict_from_row, axis=1)
        return pd.concat([test_df['id'], predictions], axis=1)

