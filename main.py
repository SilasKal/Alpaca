from llama_cpp import Llama


def llama_model(user_input):
    llm = Llama(model_path='/home/stud_homes/s9956385/models/ggml-model-q4_0.bin')
    print('Model loaded')
    output = llm(
        'Question:' + user_input+  'Answer:',
        max_tokens=100,
        temperature=0.8,
        stop=[],
        echo=True,
    )
    return output

def alpaca_model(user_input):
    llm = Llama(model_path='/home/stud_homes/s9956385/models/ggml-alpaca-7b-q4.bin')
    print('Model loaded')
    output = llm(
        'Question:' + user_input+  'Answer:',
        max_tokens=100,
        temperature=0.8,
        stop=[],
        echo=True,
    )
    return output
def vicuna_model(user_input):
    llm = Llama(model_path='/home/stud_homes/s9956385/models/ggml-vicuna-7b-1.1-q4_0.bin')
    print('Model loaded')
    output = llm(
        'Question:' + user_input+  'Answer:',
        max_tokens=100,
        temperature=0.8,
        stop=[],
        echo=True,
    )
    return output


alpaca_model('Name 10 words that are as different from each other as possible.')
