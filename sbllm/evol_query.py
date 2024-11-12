import os
import re
import sys
import time
import fcntl
import openai
import random
import logging
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from time import sleep
from copy import deepcopy
from fastbm25 import fastbm25
import multiprocessing
import google.generativeai as genai
llama_api_keys = [
"xxxxxxxxxxxxxxxxxxxx",
]
openai_api_keys = [
'xxxxxxxxxxxxxxxxxxxx',
]
gemini_api_keys = [
"xxxxxxxxxxxxxxxxxxxx"
]
api_keys = []


logger = logging.getLogger(__name__)


def clean(text):
    text = re.sub('[\u4e00-\u9fa5]+', '', text)
    text = re.sub('[\u3040-\u309F]+', '', text)
    text = re.sub('[\u30A0-\u30FF]+', '', text)
    text = re.sub('[\uAC00-\uD7A3]+', '', text)
    return text


format = '''Task Desctiption & Instructions: 
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```python\n<code>\n```

'''

format_pattern = '''Task Desctiption & Instructions: 
You will be provided with a code snippet, its existing optimization versions along with their performance, and two code transformation patterns. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```python\n<code>\n```

'''

format_ablation = '''
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks  (```python ```).". 
'''

format_ablation_cpp = '''
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks  (```cpp ```).". 
'''

format_cpp = '''Task Desctiption & Instructions: 
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```cpp\n<code>\n```

'''

format_pattern_cpp = '''Task Desctiption & Instructions: 
You will be provided with a code snippet, its existing optimization versions along with their performance, and two code transformation patterns. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```cpp\n<code>\n```

'''


def prompt_construction(cfg, queries):
    results = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')) as f:
        for obj in f:
            results.append(obj)

    prompt = []
    ids = set()
    assert len(queries) == len(results)
    for query, pperf in zip(queries, results):
        if pperf['stop'] >= 1:
            continue
        ids.add(query['idx'])
        previous_performance = ''
        for idx in range(len(pperf['best_candidates'][:cfg.beam_number])):
            opt = pperf['best_candidates'][idx]
            if opt['acc'] == 1 and opt['input_time']/opt['time']>1:
                previous_performance += 'A correct and optimized version {}:'.format(str(idx+1))
            elif opt['acc'] == 1 and opt['input_time']/opt['time']<=1:
                previous_performance += 'A correct but unoptimized version {}:'.format(str(idx+1))
            elif opt['acc'] < 1:
                previous_performance += 'An incorrect optimized version {}:'.format(str(idx+1))
            previous_performance += '\n{}\nAccuracy: {} Speedup Rate: {}\n'.format(opt['content'].strip(), str(opt['acc']), str(round(opt['input_time']/opt['time'], 2)))
        if cfg.model_name == 'gemini':
            if 'pattern' in pperf and len(pperf['pattern'])>0:
                patterns = 'Pattern 1:\n{}\nPattern 2:\n{}\n'.format(pperf['pattern'][0], pperf['pattern'][1])
                if cfg.lang=='python':
                    query['prompt_chat'] = [
                        {"role": "user", "parts": format_pattern+"The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns)},
                        ]
                else:
                    query['prompt_chat'] = [
                        {"role": "user", "parts": format_pattern_cpp+"The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns)},
                        ]
            else:
                if cfg.lang=='python':
                    query['prompt_chat'] = [
                        {"role": "user", "parts": format+"The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)},
                        ]
                else:
                    query['prompt_chat'] = [
                        {"role": "user", "parts": format_cpp+"The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)},
                        ]
        else:
            if 'pattern' in pperf and len(pperf['pattern'])>0:
                patterns = 'Pattern 1:\n{}\nPattern 2:\n{}\n'.format(pperf['pattern'][0], pperf['pattern'][1])
                if cfg.lang=='python':
                    query['prompt_chat'] = [
                        {"role": "user", "content": format_pattern+"The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns)},
                        ]
                else:
                    query['prompt_chat'] = [
                        {"role": "user", "content": format_pattern_cpp+"The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns)},
                        ]
            else:
                if cfg.lang=='python':
                    query['prompt_chat'] = [
                        {"role": "user", "content": format+"The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)},
                        ]
                else:
                    query['prompt_chat'] = [
                        {"role": "user", "content": format_cpp+"The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)},
                        ]
        prompt.append(query)
    
    return ids, prompt



def openai_llama(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    model = "codellama/CodeLlama-34b-Instruct-hf"
    openai.api_key = api_keys[0]
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    multi_fail_count = 0

    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        fail_count = 0
        query['prediction'] = []
        query['detailed_prediction'] = []
        while len(query['prediction'])<5:
            messages = [
                {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the instructions and output format specification to generate a more efficient code. The improved code should be in code blocks (```{} ```).".format(cfg.lang)},
            ]
            messages.extend(query['prompt_chat'])

            try:
                if len(query['prediction']) == 4:
                    response = openai.ChatCompletion.create(model=model, messages=messages, n=1, temperature=0.7)
                else:
                    response = openai.ChatCompletion.create(model=model, messages=messages, n=2, temperature=0.7)
                query['prediction'].extend([gen['message']['content'].strip() for gen in response["choices"]])
                query['detailed_prediction'].extend([gen for gen in response["choices"]])
                if len(query['prediction'])>=5:
                    if 'prompt_chat' in query:
                        del query['prompt_chat']
                    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(cfg.api_idx),str(len(api_keys)), 'test')), "a") as f:
                        f.write_all([query])
                    sleep(2)
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'insufficient balance' in info:
                    sys.exit()
                elif 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                elif 'Please reduce the length of the messages.' in info:
                    continue
                sleep(2)
            if fail_count>10:
                print(pos, multi_fail_count)
                multi_fail_count+=1
                break



def openai_gemini(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    genai.configure(api_key=api_keys[cfg.api_idx]) 
    generation_config = { 
        "temperature": 0.7,
        "candidate_count": 1, 
    }
    safety_settings = [ 
        {
            "category": "HARM_CATEGORY_HARASSMENT", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
        }, 
        { 
            "category": "HARM_CATEGORY_HATE_SPEECH", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
        }, 
        { 
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
        }, 
        { 
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
        } 
    ] 
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings) 
    multi_fail_count = 0
    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        fail_count = 0
        query['prediction'] = []
        while len(query['prediction'])<5:
            messages = [
                {"role": "user", "parts": "You are a software developer and now you will help to improve code efficiency. Please follow the instructions and output format specification to generate a more efficient code. The improved code should be in code blocks (```{} ```).".format(cfg.lang)},
                {'role':'model','parts':['OK, I will follow the instructions and help you improve code efficiency.']}
            ]
            messages.extend(query['prompt_chat'])

            try:
                response = model.generate_content(messages)
                query['prediction'].append(response.text)
                if len(query['prediction'])>=5:
                    if 'prompt_chat' in query:
                        del query['prompt_chat']
                    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(cfg.api_idx),str(len(api_keys)), 'test')), "a") as f:
                        f.write_all([query])
                    sleep(random.uniform(1,3))
                else:
                    sleep(random.uniform(1,2))
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'insufficient balance' in info:
                    sys.exit()
                elif 'Max retries exceeded with url:' in info:
                    sleep(random.uniform(1,3)*fail_count)
                elif 'Please reduce the length of the messages.' in info:
                    continue
                print(info)
                sleep(random.uniform(1,3))
            if fail_count>10:
                print(pos, multi_fail_count)
                multi_fail_count+=1
                break

def openai_official(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    multi_fail_count = 0
    model = "gpt-3.5-turbo-0613"
    openai.api_key = api_keys[cfg.api_idx]
    exceed_time = 0
    print(openai.api_key)

    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        success = 0
        fail_count = 0
        while success!=1:
            messages = [
                {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the instructions and output format specification to generate a more efficient code. The improved code should be in code blocks (```{} ```).".format(cfg.lang)},
            ]
            messages.extend(query['prompt_chat'])
            try:    
                response = openai.ChatCompletion.create(model=model, messages=messages, n=cfg.generation_number, temperature=0.7)
                success=1
                model = "gpt-3.5-turbo-0613"
                query['prediction'] = [response["choices"][num]['message']['content'].strip() for num in range(cfg.generation_number)]
                query['detailed_prediction'] = response["choices"]
                if 'prompt_chat' in query:
                    del query['prompt_chat']
                with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(cfg.api_idx),str(len(api_keys)), 'test')), "a") as f:
                    f.write_all([query])
                sleep(0.02)
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'insufficient balance' in info:
                    sys.exit(-1)
                elif 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                elif 'Please reduce the length of the messages.' in info:
                    model = "gpt-3.5-turbo-16k-0613"
                    exceed_time+=1
                print(info)
            if fail_count>10:
                print(pos, multi_fail_count)
                multi_fail_count+=1
                break
    print('exceed_time: ',exceed_time)



def read_file(cfg):
    queries = []
    exist_queries=set()
    with jsonlines.open(cfg.baseline_data_path) as reader:
        for obj in reader:
            queries.append(obj)
    queries = queries[int(len(queries)*(cfg.slice-1)/cfg.total):int(len(queries)*cfg.slice/cfg.total)]

    if cfg.iteration>0:
        ids, queries = prompt_construction(cfg, queries)
    else:
        ids = [i for i in range(1000)]
    print('{} codes will be optimized in iteration {}'.format(len(queries), cfg.iteration))

    unsort_queries = []
    for num in range(len(api_keys)):
        if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))):
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))) as f:
                for obj in f:
                    unsort_queries.append(obj)
        if len(unsort_queries)>0:
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test')), 'w') as f:
                f.write_all(unsort_queries)

    if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
            for obj in f:
                exist_queries.add(obj['idx'])
        new_queries = []
        for query in queries:
            if query['idx'] not in exist_queries:
                new_queries.append(query)
        queries = new_queries
        
    return ids, queries, len(exist_queries)

def separate_lines(code):
    lines = code.split('\n')
    comments = []
    code_lines = []
    start = False

    for line in lines:
        if not start and line.startswith('#'):
            comments.append(line[1:].strip())
        else:
            start=True
            code_lines.append(line)

    return ' '.join(comments), '\n'.join(code_lines)




def post_process(cfg, ids):
    data = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
        for obj in f:
            data.append(obj)
    count=0
    
    processed_data = []
    for obj in data:
        if obj['idx'] not in ids:
            continue
        if cfg.lang == 'python':
            post_prediction = []
            detailed_prediction = []
            for res in obj['prediction']:
                detailed_prediction.append(res)
                if res.count('```python') == 1:
                    post_prediction.append(res.split('```python')[1].split('```')[0])
                elif res.count('``` python') == 1:
                    post_prediction.append(res.split('``` python')[1].split('```')[0])
                elif res.count('```Python') == 1:
                    post_prediction.append(res.split('```Python')[1].split('```')[0])
                elif res.count('```') == 2:
                    post_prediction.append(res.split('```')[1])
                elif res.startswith('#'):
                    r, c = separate_lines(res)
                    post_prediction.append(c)
                elif res.count('```python') > 1:
                    post_prediction.append(res.split('```python')[-1].split('```')[0])
                else:
                    post_prediction.append(res)
        else:
            post_prediction = []
            detailed_prediction = []
            for res in obj['prediction']:
                detailed_prediction.append(res)
                if res.count('```cpp') == 1:
                    post_prediction.append(res.split('```cpp')[1].split('```')[0])
                elif res.count('``` cpp') == 1:
                    post_prediction.append(res.split('``` cpp')[1].split('```')[0])
                elif res.count('```Cpp') == 1:
                    post_prediction.append(res.split('```Cpp')[1].split('```')[0])
                elif res.count('```') == 2:
                    post_prediction.append(res.split('```')[1])
                elif res.startswith('#'):
                    r, c = separate_lines(res)
                    post_prediction.append(c)
                elif res.count('```cpp') > 1:
                    post_prediction.append(res.split('```cpp')[-1].split('```')[0])
                else:
                    post_prediction.append(res)
        obj['prediction'] = post_prediction
        if 'detailed_prediction' not in obj:
            obj['detailed_prediction'] = detailed_prediction
        processed_data.append(obj)
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(processed_data)
    print(count)
    for num in range(len(api_keys)):
        if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))):
            os.remove(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_file", action='store_true')
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--generation_path", default=None, type=str, required=True)
    parser.add_argument("--baseline_data_path", default='None', type=str)
    parser.add_argument("--training_data_path", default='None', type=str)
    parser.add_argument("--lang", default='None', type=str)
    parser.add_argument("--api_idx", default=0, type=int)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--restart_pos", default=0, type=int)
    parser.add_argument("--beam_number", default=1, type=int)
    parser.add_argument("--generation_number", default=1, type=int)

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--total", default=1, type=int)
    cfg = parser.parse_args()
    if not os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration))):
        os.makedirs(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration)))
    fail_count = 0
    if cfg.model_name == 'gemini':
        api_keys = gemini_api_keys
    elif cfg.model_name == 'codellama':
        api_keys = llama_api_keys
    else:
        api_keys = openai_api_keys
    while True:
        ids, queries, previous_length = read_file(cfg)
        print(previous_length)
        if cfg.model_name == 'gemini' or cfg.model_name == 'codellama':
            post_process(cfg, ids)
            sys.exit(0)

        pool = multiprocessing.Pool(len(api_keys))
        chunks_query = np.array_split(queries, len(api_keys))
        chunks_data = []
        print('Number of queries in round {}: {}'.format(str(fail_count), str(len(queries))))
        for i in range(len(api_keys)):
            tmp_data={}
            tmp_data['cfg'] = deepcopy(cfg)
            tmp_data['cfg'].api_idx = i
            tmp_data['queries'] = chunks_query[i]
            chunks_data.append(tmp_data)
        if cfg.model_name == 'gemini':
            results = pool.map(openai_gemini, chunks_data)
        elif cfg.model_name == 'codellama':
            results = pool.map(openai_llama, chunks_data)
        else:
            results = pool.map(openai_official, chunks_data)

        data = []
        if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
                for i in f:
                    data.append(i)
        
        if len(data) != len(queries)+previous_length and fail_count>5:
            sys.exit(-1)
        elif len(data) != len(queries)+previous_length:
            fail_count+=1
        else:
            post_process(cfg, ids)
            sys.exit(0)



