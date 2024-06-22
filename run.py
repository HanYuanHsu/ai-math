from deploy.deepseek_math_api import DeepseekMath
from prompts.prompt import CODE1
from utils.ours import *

MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"
HF_CACHE_DIR = "/home/master/13/hhhsu/models"

def solve(problem: str):
    model = DeepseekMath(model_name=MODEL_NAME,
                         hf_cache_dir=HF_CACHE_DIR, # still downloads shards?
                         devices=2,
                         quant=True)
    
    model_input = CODE1.format(problem, "{}")
    model_output = model.generate(model_input)

    scripts = extract_python_blocks(model_output)
    if len(scripts) < 1:
        raise Exception("no scripts in output")
    for code in scripts:
        pass

    print(f"model_output: {model_output}")

if __name__ == "__main__":
    problem = "Solve $\log_4 x + \log_2 x^2 = 10$."
    solve(problem)
