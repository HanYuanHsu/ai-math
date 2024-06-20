'''

This Flask application serves 2 api endpoints for the remote notebook
to retrieve the data, submit the answer and we will return retu
rn the 
evaluation results for the remote notebook.

- /problem/next
    - description
        - get the next problem's information. the remote notebook can
          use response['problem']['problem'] to start its prediction.
    - method: GET
    - data
        - None
    - response
        - 'problem': return a problem object

- /problem/<index>/submit
    - description
        - the remote notebook can submit its LLM response & predicted
          number(after mod operation).
        - if the index is the last problem's index, you can get the
          evaluation result from the response (see the data part)
    - method: POST
    - data
        - response
            - LLM's response
        - predict_answer
            - a integer number ( the answer from LLM mod by 1000 )
    - response
        - status
            - 'ok' of 'fail'
        - msg
            - error message or 'correct' or 'incorrect'
        - finished
            - bool. means whether the evaluation is finished.
              If true, you can get the evaluation results from 'result'
        - result
            - if 'finished' is False, this will be 'None'
            - if 'finished' is True, you will get a 'str'
              

          

'''
from utils.file import load_json, dump_json
from datasets.MATH import MATHDataset
from models.base import mod1000
from flask import Flask, request
import json
import os
import argparse

MODEL = "deepseek"
HOST = '0.0.0.0'
PORT = 3389

PROBLEM_INDEX = 0
DATASET = None
PROBLEM_SET_JSON = {}

CORRECT = 0
CORRECT_CATEGORIES = {}

app = Flask(__name__)

def calculate_accuracy_from_json(json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)
    n_data = len(data)
    n_correct = 0
    for d in data:
        if d["status"] == "correct":
            n_correct += 1
        elif not (d["status"] == "incorrect"):
            raise Exception('Problem with d["status"]')
    print(f"number of data: {n_data}")
    print(f"number of correct predictions: {n_correct}")
    print(f"accuracy: {n_correct / n_data}")


@app.route('/problem/next', methods=['GET'])
def next_problem():
    global PROBLEM_INDEX
    if PROBLEM_INDEX != len(DATASET):
        print(f"remote notebook get the problem with index={PROBLEM_INDEX}")
        problem = DATASET[PROBLEM_INDEX]
        PROBLEM_INDEX += 1

    else:
        print(f"finished. index={PROBLEM_INDEX} now")
        problem = None
        
    return json.dumps({'problem': problem})

@app.route('/problem/<int:index>/submit', methods=['POST'])
def submit(index):
    global CORRECT
    global CORRECT_CATEGORIES

    if index != PROBLEM_INDEX-1:
        return json.dumps({'status': 'fail', 'msg': f'you should not submit problem {index} answer'})
    else:
        problem = DATASET[index]
        response, predict_answer = request.values.get('response'), int(request.values.get('predict_answer'))
        PROBLEM_SET_JSON[index]['response'] = response
        PROBLEM_SET_JSON[index]['predict'] = predict_answer
        PROBLEM_SET_JSON[index]['status'] = 'incorrect'
        target = mod1000(problem['answer'])
        if predict_answer == target:
            PROBLEM_SET_JSON[index]['status'] = 'correct'
            CORRECT_CATEGORIES[problem['type']] = 1 if problem['type'] not in CORRECT_CATEGORIES else CORRECT_CATEGORIES[problem['type']] + 1
            CORRECT += 1

        response = {'status': 'ok', 'msg': PROBLEM_SET_JSON[index]['status'], 'finished': False, 'result': None}
        if index == len(DATASET)-1:        # this submition is for the last problem, so we need to print out evaluation result.
            response['finished'] = True

            result_msg = ""
            result_msg += f"Correct: {CORRECT}/{len(DATASET)} ({CORRECT/len(DATASET)}%)\n"
            result_msg += '\n'.join([ f"{category}: {correct_num}/10 ({correct_num}%)" for category, correct_num in CORRECT_CATEGORIES.items() ])
            response['result'] = result_msg

            dump_json(PROBLEM_SET_JSON, 'dataset_predicted.json')
        return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)

    DATASET = MATHDataset('./datasets/MATH', dataset_type='train', sample_per_categorie_num=10)
    PROBLEM_SET_JSON = load_json('./dataset.json')

    args = parser.parse_args()
    #print(f"host: {args.host}")

    app.run(host=args.host, port=8888)
    
