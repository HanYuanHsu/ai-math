https://chat.openai.com/share/46f3ed0f-32fe-4fcf-8c6a-2dd85c586d78

You are a mathematician that is excellent at finding patterns and trends from numbers. For example, given the sequence 2, 4, 6, 8, 10, ..., you should say that sequence is increasing. Now, given 1.3, 1.5, 1.7, 1.9, what can you say

## Run Evaluation

Take DeepSeek model for example:

1. Deploy the model

    Default run on http://localhost:8888

    ```
    cd deploy && python deepseek.py
    ```
2. Evaluate
    
    You can change the prompt in `prompts/prompt.py`

    ```
    python evaluate/deepseek.py
    ```
