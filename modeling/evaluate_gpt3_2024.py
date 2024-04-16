import os
import numpy as np
import operator
import json
from dataset.util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv

import openai
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise Exception("openai api key is None")
openai.api_key = api_key

client = OpenAI()

def call_engine(problem):
    # few-shot

    response = client.chat.completions.create(
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
            {"role": "user", "content": problem}
        ]
    )

    return response.choices[0].message.content

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


rootdir = "data/test" 

def run(max=-1):
    outputs = []
    answers = []
    types = []
    levels = []

    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fnames_list.append(os.path.join(subdir, file))
            with open(os.path.join(subdir, file), 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {file}", e)
                    raise e
                prob_level = problem_data["level"]
                prob_type = problem_data["type"]
                try:
                    prob_level = int(prob_level.split("Level ")[1])
                except:
                    prob_level = None
                model_output = call_engine(problem_data["problem"])
                answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))

                levels.append(prob_level)
                types.append(prob_type)
                outputs.append(model_output)
                answers.append(answer)

                print("Model output:")
                print(model_output)
                print("Correct answer:")
                print(answer)
                print("--------------------------------------------")

                try:
                    equiv = is_equiv(model_output, answer)
                except:
                    equiv = False
                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(equiv)
                else:
                    cors[(prob_level, prob_type)] = [equiv]
                if prob_level in level_cors:
                    level_cors[prob_level].append(equiv)
                else:
                    if prob_level is not None:
                        level_cors[prob_level] = [equiv]
                if prob_type in subject_cors:
                    subject_cors[prob_type].append(equiv)
                else:
                    if prob_type is not None:
                        subject_cors[prob_type] = [equiv]
                if equiv:
                    correct += 1
                total += 1

                print(str(correct) + "/" + str(total))

            if max > 0 and total > max:
                break
        if max > 0 and total > max:
            break

    with open("outputs_answers_gpt3_fewshot.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))

if __name__ == "__main__":
    run(max=10)

    # for testing:
    # for engine in ["ada"]:
    #     run(engine, max=10)
