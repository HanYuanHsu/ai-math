import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from flask import Flask, request

MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Deepseek():
    def __init__(self, model_name, device): 
        self.device = device

        print(f"Load {model_name} to {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def predict(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_tensor, max_new_tokens=512)

        return self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

app = Flask(__name__)
model = Deepseek(MODEL_NAME, DEVICE)

@app.route("/", methods=['POST'])
def receive():
    prompt = request.values.get('prompt')
    response = model.predict(prompt)
    return {'response': response}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)

