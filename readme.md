https://chat.openai.com/share/46f3ed0f-32fe-4fcf-8c6a-2dd85c586d78

You are a mathematician that is excellent at finding patterns and trends from numbers. For example, given the sequence 2, 4, 6, 8, 10, ..., you should say that sequence is increasing. Now, given 1.3, 1.5, 1.7, 1.9, what can you say

I have downloaded code llama weights in cml7. Check contents of /tmp2/hhhsu <-- cml7 gone
Go to cml9

## Run Evaluation

Take DeepSeek model for example:

1. Deploy the model

    Default run on http://localhost:8888

    ```
    cd deploy && python deepseek_api.py
    ```
2. Evaluate
    
    You can change the prompt in `prompts/prompt.py`

    ```
    python evaluate/deepseek.py
    ```

## Add models

You will complete these `python` files.

- `deploy/${NEW_MODEL}.py`: how to deploy your model to handle the prompt requests
- `prompts/prompt.py`: Add your prompt for your new models
- `models/${NEW_MODEL}.py`: how to request to your model which deployed
- `evaluate.py`: import your model and use
