import os
import re
import tokenize
from io import StringIO
import yaml
import time
import jsonlines 
import subprocess
import jsonlines


def write_yaml(data, file_path):
    with open(file=file_path,mode='w',encoding='utf8') as f:
        yaml.dump(data,f)


def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)




def testing_and_reporting(cfg):
    print('mapping......')
    input_code_map = {}
    with jsonlines.open("../processed_data/{}/test.jsonl".format(cfg.lang)) as f:
        for obj in f:
            input_code_map[remove_comments_and_docstrings(obj['code_v0_no_empty_lines'], cfg.lang)] = obj['input']

    print('processing......')
    processed = []
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'tmp_queries_{}.jsonl'.format(cfg.model_name+'test'))) as f:
        for i in f:
            processed.append({'slow_code_col':input_code_map[i['query']], 'model_generated_potentially_faster_code_col':i['prediction']})
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.jsonl'.format(cfg.model_name+'test')),'w') as f:
        f.write_all(processed)

    data = {}
    data['language'] = cfg.lang
    data['model_generated_outputs_path'] = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.jsonl'.format(cfg.model_name+'test'))
    data['inputs_outputs_basepath'] = cfg.test_case_path
    data['reference_file_path'] = "../processed_data/{}/test.jsonl".format(cfg.lang)
    data['output_report_file_path'] = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))
    data['preprocessed_output_file'] = "../processed_data/{}/processed_time.reports".format(cfg.lang)
    data['num_problems_to_evaluate'] = -1
    data['num_trials'] = 8
    data['ignore_first_k'] = 1
    data['max_time_per_run'] = 10
    data['temp_dir'] = "../processed_data/{}/generated_tmp".format(cfg.lang)
    data['model_generated_potentially_faster_code_col'] = "model_generated_potentially_faster_code_col"
    data['slow_code_col'] = "slow_code_col"
    data['reference_code_col'] = "target"
    data['reference_input_col'] = "input"
    data['is_prompt_based'] = False
    data['run_reference'] = True
    data['run_input'] = True
    data['cpu_number'] = 1
    data['process_number'] = int(cfg.process_number)
    write_yaml(data, os.path.join(cfg.output_path, cfg.mode, 'eval_config.yaml'))

    error_file = open(os.path.join(cfg.output_path, cfg.mode, "stderr.txt"), "wb")
    out_file = open(os.path.join(cfg.output_path, cfg.mode, "output.txt"),"wb")
    print("testing......")
    if not os.path.exists(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))):
        cmd = 'cd ../pie; python src/codenet_eval/run_eval_feedback.py --eval_config {}'.format(os.path.join(cfg.output_path, cfg.mode, 'eval_config.yaml'))
        child = subprocess.Popen(cmd, shell=True, stdout=out_file, stderr=error_file, bufsize=-1, start_new_session=True)
        while True:
            Flag = child.poll()
            if Flag == 0:
                error_file.close()
                out_file.close()
                break
            else:
                time.sleep(10)


    results = []
    references = []
    hypothesis = []
    ptr = 0
    correct = 0
    faster_count = 0
    unique_count = 0
    input_time_sum = 0
    generated_test_sum = 0
    unique_reference_time_sum = 0
    unique_generated_test_sum = 0
    execution_data = []
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))) as f:
        for i in f:
            execution_data.append(i)

    for i in execution_data:
        acc = i['model_generated_potentially_faster_code_col_acc']
        input_time = i['input_time_mean']
        generated_time = i['model_generated_potentially_faster_code_col_time_mean']
        reference_time = i['reference_time_mean']
        if input_time  is None or reference_time is None:
            continue
        if generated_time is None:
            generated_time = input_time
        results.append([generated_time, input_time, acc])
        for num in range(5):
            if i['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(num))]==i['model_generated_potentially_faster_code_col_time_mean'] and \
               i['model_generated_potentially_faster_code_col_{}_time_std'.format(str(num))]==i['model_generated_potentially_faster_code_col_time_std'] :
                references.append([i['code_v1_no_empty_lines']])
                hypothesis.append(i['model_generated_potentially_faster_code_col_{}'.format(str(num))])
                break
        if acc==1:
            correct+=1
        if acc==1 and generated_time<input_time:
            if generated_time<reference_time:
               unique_count+=1
               unique_reference_time_sum += reference_time
               unique_generated_test_sum += generated_time
            if generated_time<input_time*0.9:
                faster_count+=1
            input_time_sum += input_time
            generated_test_sum += generated_time
            ptr += input_time/generated_time -1
        else:
            input_time_sum += input_time
            generated_test_sum += input_time
            ptr += 0
    print(cfg.mode)
    print('OPT(%): ', round(100*faster_count/len(results), 2))
    print('SP: ', round(100*ptr/len(results), 2))

