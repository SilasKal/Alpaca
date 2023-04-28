import json

from llama_cpp import Llama

for i in ['ggml-alpaca-7b-q4.bin', 'ggml-model-q4_0.bin', 'ggml-vicuna-7b-1.1-q4_0.bin',  'gpt4all-lora-quantized.bin']:
    print('Loading model' + i)
    llm = Llama(model_path='/home/stud_homes/s9956385/models/' + i)
    print('Model loaded')
    output = llm(
        'Question: Name 10 words that are as different from each other as possible? Answer:',
        max_tokens = 100,
        temperature=0.8,
        stop= ['\n', 'Question:', 'Q:'],
        echo = True,
    )

    print(output)
