import os
import re
import sys
import time
import openai
import random
import logging
import argparse
import jsonlines
import numpy as np
import multiprocessing
from tqdm import tqdm
from time import sleep
from copy import deepcopy
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

                


def read_file(cfg):
    queries = []
    exist_queries = set()
    with jsonlines.open(cfg.baseline_data_path) as reader:
        for obj in reader:
            queries.append(obj)

    unsort_queries = []
    for num in range(len(api_keys)):
        if os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))):
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))) as f:
                for obj in f:
                    if 'detailed_prediction' in obj:
                        del obj['detailed_prediction']
                    unsort_queries.append(obj)
        if len(unsort_queries)>0:
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test')), 'w') as f:
                f.write_all(unsort_queries)

    if os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
            for obj in f:
                exist_queries.add(obj['idx'])
        new_queries = []
        for query in queries:
            if query['idx'] not in exist_queries:
                new_queries.append(query)
        queries = new_queries
        
    return queries, len(exist_queries)



def openai_llama(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    model = "codellama/CodeLlama-34b-Instruct-hf"
    openai.api_key = api_keys[cfg.api_idx]
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    multi_fail_count = 0
    print(openai.api_key)


    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        fail_count = 0
        query['prediction'] = []
        query['detailed_prediction'] = []
        stop = False
        while not stop:
            if 'cot' in cfg.mode:
                messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the output format of the examples and Please reply with the improved code in code blocks (```{} ```) and explain the reasons briefly at the beginning.".format(cfg.lang)},
                    ]
            else:
                messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Only reply with the source code. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks  (```{} ```).".format(cfg.lang)},
                    ]
            length = 0
            for i, exemplar in enumerate(query['prompt_chat']):
                messages.append({"role":"user","content": '# Example {}\n'.format(str(i))+exemplar['query'].replace("# Let's think step by step.",'')})
                length+=len(exemplar['query'].split())
                if 'response' in exemplar:
                    messages.append({"role":"assistant","content": exemplar['response']})
                    length+=len(exemplar['response'].split())

            try:
                response = openai.ChatCompletion.create(model=model, messages=messages, n=1, temperature=0.7)
                query['prediction'].extend([gen['message']['content'].strip() for gen in response["choices"]])
                if len(query['prediction'])>=10 or (length>2600 and len(query['prediction'])>=5):
                    stop=True
                    if 'prompt_chat' in query:
                        del query['prompt_chat']
                    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name, str(cfg.api_idx), str(len(api_keys)), 'test')), "a") as f:
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
                print(info)
                sleep(2)
            if fail_count>10:
                print(pos, multi_fail_count)
                multi_fail_count+=1
                break



def openai_gemini(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']


    genai.configure(api_key=api_keys[cfg.api_idx]) 
    print(api_keys[cfg.api_idx])
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
            if 'cot' in cfg.mode:
                messages = [
                    {"role": "user", "parts": ["You are a software developer and now you will help to improve code efficiency. Please follow the output format of the examples and Please reply with the improved code in code blocks (```{} ```) and explain the reasons briefly at the beginning.".format(cfg.lang)]},
                    {'role':'model','parts':['OK, I will help you improve code efficiency.']}
                    ]
            else:
                messages = [
                    {"role": "user", "parts": ["You are a software developer and now you will help to improve code efficiency. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks  (```{} ```).".format(cfg.lang)]}, 
                    {'role':'model','parts':['OK, I will help you improve code efficiency.']}
                    ]

            for i, exemplar in enumerate(query['prompt_chat']):
                messages.append({"role":"user","parts": ['# Example {}\n'.format(str(i))+exemplar['query']]})
                if 'response' in exemplar:
                    messages.append({"role":"model","parts": [exemplar['response']]})

            try:
                response = model.generate_content(messages)
                query['prediction'].append(response.text)
                if len(query['prediction'])>=5:
                    if 'prompt_chat' in query:
                        del query['prompt_chat']
                    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name, str(cfg.api_idx), str(len(api_keys)), 'test')), "a") as f:
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
    #model = "gpt-3.5-turbo-0613"
    model = "gpt-4-1106-preview"
    openai.api_key = api_keys[cfg.api_idx]
    print(openai.api_key)


    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        success = 0
        fail_count = 0
        while success!=1:
            if 'cot' in cfg.mode:
                messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the output format of the examples. Please reply with the improved code in code blocks (```{} ```) and explain the reasons briefly at the beginning.".format(cfg.lang)},
                    ]
            else:
                messages = [
                    {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Only reply with the source code. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks  (```{} ```).".format(cfg.lang)},
                    ]

            for i, exemplar in enumerate(query['prompt_chat'][:3]+query['prompt_chat'][-1:]):
                messages.append({"role":"user","content": '# Example {}\n'.format(str(i))+exemplar['query']})
                if 'response' in exemplar:
                    messages.append({"role":"assistant","content": exemplar['response']})

            try:
                response = openai.ChatCompletion.create(model=model, messages=messages, n=cfg.generation_number, temperature=0.7,logprobs=1)
                success=1
                #model = "gpt-3.5-turbo-0613"
                model = "gpt-4-1106-preview"
                probs = []
                for i in response["choices"]:
                    probs.append(sum([con['logprob'] for con in i['logprobs']["content"]]))
                order = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
                query['prediction'] = [response["choices"][num]['message']['content'].strip() for num in order] 
                query['detailed_prediction'] = response["choices"]
                if 'prompt_chat' in query:
                    del query['prompt_chat']
                with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name, str(cfg.api_idx), str(len(api_keys)), 'test')), "a") as f:
                    f.write_all([query])
                sleep(0.01)
            except Exception  as e:
                info = e.args[0]
                fail_count+=1
                if 'insufficient balance' in info:
                    api_idx+=1
                    openai.api_key = api_keys[api_idx]
                elif 'Max retries exceeded with url:' in info:
                    sleep(2*fail_count)
                elif 'Please reduce the length of the messages.' in info:
                    model = "gpt-3.5-turbo-16k-0613"
                    continue
                print(info)
            if fail_count>10:
                print(pos, multi_fail_count)
                multi_fail_count+=1
                break


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


def post_process(cfg):
    data = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test')), "r") as f:
        for obj in f:
            data.append(obj)
    count=0
    processed_data = []
    for obj in data:
        if cfg.lang == 'python':
            post_prediction = []
            for res in obj['prediction']:
                if res.count('```python') == 1:
                    post_prediction.append(res.split('```python')[1].split('```')[0])
                elif res.count('``` python') == 1:
                    post_prediction.append(res.split('``` python')[1].split('```')[0])
                elif res.count('```Python') == 1:
                    post_prediction.append(res.split('```Python')[1].split('```')[0])
                elif res.count('```None') == 1:
                    post_prediction.append(res.split('```None')[1].split('```')[0])
                elif res.count('```') == 2:
                    post_prediction.append(res.split('```')[1])
                elif res.count('```python') > 1:
                    post_prediction.append(res.split('```python')[-1].split('```')[0])
                elif res.startswith('#'):
                    r, c = separate_lines(res)
                    post_prediction.append(c)
                else:
                    post_prediction.append(res)
                    count+=1
        else:
            post_prediction = []
            for res in obj['prediction']:
                if res.count('```cpp') == 1:
                    post_prediction.append(res.split('```cpp')[1].split('```')[0])
                elif res.count('``` cpp') == 1:
                    post_prediction.append(res.split('``` cpp')[1].split('```')[0])
                elif res.count('```Cpp') == 1:
                    post_prediction.append(res.split('```Cpp')[1].split('```')[0])
                elif res.count('```None') == 1:
                    post_prediction.append(res.split('```None')[1].split('```')[0])
                elif res.count('```') == 2:
                    post_prediction.append(res.split('```')[1])
                elif res.startswith('#'):
                    r, c = separate_lines(res)
                    post_prediction.append(c)
                elif res.count('```cpp') > 1:
                    post_prediction.append(res.split('```cpp')[-1].split('```')[0])
                else:
                    post_prediction.append(res)
                    count+=1
        obj['prediction'] = post_prediction
        processed_data.append(obj)
    print(count)

    final_results = []
    for i in data:
        final_results.append({'query': i['query'], 'prediction': i['prediction'][:1]})
    if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'top1')):
        os.makedirs(os.path.join(cfg.generation_path, cfg.mode, 'top1'))
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'top1', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(final_results)

    final_results = []
    for i in data:
        final_results.append({'query': i['query'], 'prediction': i['prediction'][:3]})
    if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'top3')):
        os.makedirs(os.path.join(cfg.generation_path, cfg.mode, 'top3'))
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'top3', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(final_results)

    final_results = []
    for i in data:
        final_results.append({'query': i['query'], 'prediction': i['prediction'][:20]})
    if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'top5')):
        os.makedirs(os.path.join(cfg.generation_path, cfg.mode, 'top5'))
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'top5', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(final_results)

    for num in range(len(api_keys)):
        if os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))):
            os.remove(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test')))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--generation_path", default=None, type=str, required=True)
    parser.add_argument("--baseline_data_path", default='None', type=str)
    parser.add_argument("--lang", default='cpp', type=str)
    parser.add_argument("--generation_number", default=5, type=int)
    cfg = parser.parse_args()
    if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode)):
        os.makedirs(os.path.join(cfg.generation_path, cfg.mode))
    fail_count = 0
    if cfg.model_name == 'gemini':
        api_keys = gemini_api_keys
    elif cfg.model_name == 'codellama':
        api_keys = llama_api_keys
    else:
        api_keys = openai_api_keys
    while True:
        queries, previous_length = read_file(cfg)

        pool = multiprocessing.Pool(len(api_keys))
        chunks_query = np.array_split(queries, len(api_keys))
        chunks_data = []
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
        if os.path.exists(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
                for i in f:
                    data.append(i)
        
        if len(data) != len(queries)+previous_length and fail_count>5:
            sys.exit(-1)
        elif len(data) != len(queries)+previous_length:
            fail_count+=1
        else:
            post_process(cfg)
            sys.exit(0)

