import json

from llama_cpp import Llama

print('Loading model')
llm = Llama(model_path='/home/stud_homes/s9956385/models/ggml-alpaca-7b-q4.bin')
print('Model loaded')
output = llm(
    'Question: Name 10 words that are as different from each other as possible? Answer:',
    max_tokens = 100,
    temperature=0.8,
    stop= ['\n', 'Question:', 'Q:'],
    echo = True,
)

print(output)
