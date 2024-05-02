import sys
sys.path.append('/home/snsd0805/code/Ai-Math/ai-math/')
from utils.file import load_json, dump_json
from models.deepseek import DeepSeek
from datasets.MATH import MATHDataset
from models.base import mod1000

HOST = '127.0.0.1'
PORT = 8888

if __name__ == '__main__':
    dataset = MATHDataset('./datasets/MATH', dataset_type='train', sample_per_categorie_num=10)
    model = DeepSeek(HOST, PORT)
    problem_set_json = load_json('./dataset.json')

    correct = 0
    correct_categories = {}
    for index, i in enumerate(dataset):
        print(f"{index} problem_ID={i['id']} answer={i['answer']}")
        response, predict_answer = model.predict(i['problem'])
        print(f"    response answer: {predict_answer}")
        problem_set_json[index]['response'] = response
        problem_set_json[index]['predict'] = predict_answer
        problem_set_json[index]['status'] = 'incorrect'
        target = mod1000(i['answer'])
        if predict_answer == target:
            problem_set_json[index]['status'] = 'correct'
            correct_categories[i['type']] = 1 if i['type'] not in correct_categories else correct_categories[i['type']] + 1
            correct += 1
    print(f"Correct: {correct}/{len(dataset)} ({correct/len(dataset)}%)")
    for k, v in correct_categories.items():
        print(f"{k}: {v}/10 ({v/10}%)")
    dump_json(problem_set_json, 'dataset_predicted.json')
