from deploy.deepseek_math_api import DeepseekMath
from prompts.prompt import CODE1

MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"
HF_CACHE_DIR = "/home/master/13/hhhsu/models"

def solve(problem: str):
    model = DeepseekMath(model_name=MODEL_NAME,
                         hf_cache_dir=HF_CACHE_DIR,
                         devices=[0, 1],
                         quant=True)
    
    model_input = CODE1.format(problem, "{}")
    model_output = model.generate(model_input)

    print(f"model_output: {model_output}")
