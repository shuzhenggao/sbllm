import os
import re
import ast
import random
import argparse
import tokenize
from io import StringIO
from io import BytesIO
import jsonlines
from tqdm import tqdm
import ast
import random
import difflib
import numpy as np
import editdistance
import multiprocessing
from collections import Counter
from rank_bm25 import BM25Okapi
import builtins


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

def better(obj1, obj2):
    if obj2['acc'] == 0 and obj2['time']==99999:
        return True
    elif obj1['acc'] == 1 and obj1['input_time']/obj1['time'] > obj2['input_time']/obj2['time']:
        return True
    else:
        return False

def extract(code, content):
    if content.count('provide a new optimized code snippet:**') == 1:
        return content.split('provide a new optimized code snippet:')[1].strip()
    elif content.count('provide a new optimized code snippet:') == 1:
        return content.split('provide a new optimized code snippet:')[1].strip()
    elif content.count('provide a new optimized code snippet') == 1:
        return content.split('provide a new optimized code snippet')[1].strip()
    else:
        return '```python\n{}\n```'.format(code)


def compare_code_structure(code1, code2):
    try:
        ast1 = ast.parse(code1)
        ast2 = ast.parse(code2)
        return compare_ast_structure(ast1, ast2)
    except SyntaxError:
        return False

def compare_ast_structure(ast1, ast2):
    if type(ast1) is not type(ast2):
        return False
    if isinstance(ast1, ast.AST):
        return all(compare_ast_structure(child1, child2) for child1, child2 in zip(ast.iter_child_nodes(ast1), ast.iter_child_nodes(ast2)))
    elif isinstance(ast1, (list, tuple)):
        return all(compare_ast_structure(item1, item2) for item1, item2 in zip(ast1, ast2))
    else:
        return ast1 == ast2


class CodeAbstractor(ast.NodeTransformer):
    def __init__(self, variable_char='VAR', string_char='STR', number_char='NUM'):
        self.variable_char = variable_char
        self.string_char = string_char
        self.number_char = number_char
        self.variable_count = 0
        self.string_count = 0
        self.number_count = 0
        self.builtin_functions = set(dir(builtins))

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg.arg = f'{self.variable_char}'
            self.variable_count += 1
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        self.generic_visit(node)
        if isinstance(node.value, str):
            node.value = f'{self.string_char}'
            self.string_count += 1
        elif isinstance(node.value, (int, float, complex)):
            node.value = f'{self.number_char}'
            self.number_count += 1
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id not in self.builtin_functions:
            node.id = f'{self.variable_char}'
            self.variable_count += 1
        return node

    def visit_Str(self, node):
        self.generic_visit(node)
        node.s = f'{self.string_char}'
        self.string_count += 1
        return node

    def visit_Num(self, node):
        self.generic_visit(node)
        node.n = f'{self.number_char}'
        self.number_count += 1
        return node

def abstract_code(code):
    parsed_code = ast.parse(code)
    abstractor = CodeAbstractor()
    abstracted_ast = abstractor.visit(parsed_code)
    abstracted_code = ast.unparse(abstracted_ast)
    return abstracted_code


def tokenize_code(code):
    tokens = []
    g = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum != tokenize.ENCODING:
            tokens.append(tokval)
    return tokens

def selection(candidates, cfg):
    correct_candidates = []
    wrong_candidates = []
    for obj in candidates:
        if obj['acc']==1:
            correct_candidates.append(obj)
        else:
            wrong_candidates.append(obj)
    correct_candidates = sorted(correct_candidates, key=lambda x: (x['time']/x['input_time']))

    selected = []
    seen = []
    for i in correct_candidates:
        dup = False
        for j in seen:
            try:
                if str(abstract_code(i['code'])) == str(abstract_code(j)):
                    dup = True
                    break
            except SyntaxError:
                dup = False
        if not dup:
            selected.append(i)
            seen.append(i['code'])
    if len(selected)<cfg.beam_number:
        distances = []
        for i, obj1 in enumerate(wrong_candidates):
            code1 = obj1['code']
            distance_sum = 0
            for j, obj2 in enumerate(wrong_candidates):
                code2 = obj2['code']
                if i != j:
                    try:
                        seq1 = abstract_code(code1)
                        seq2 = abstract_code(code2)
                        distance_sum += editdistance.eval(seq1, seq2)
                    except SyntaxError:
                        distance_sum += 9999
            distances.append((i, distance_sum))
        sorted_distances = sorted(distances, key=lambda x: x[1])
        closest_segments = [wrong_candidates[index] for index, _ in sorted_distances]
        selected.extend(closest_segments)
        
    return selected

def divide_into_parts(n, k):
    quotient, remainder = divmod(n, k)
    parts = [quotient + 1 if i < remainder else quotient for i in range(k)]
    
    cumulative_parts = [0]
    cumulative_sum = 0
    for part in parts:
        cumulative_sum += part
        cumulative_parts.append(cumulative_sum)
    
    return cumulative_parts

    
def get_diff_lines(lines1, lines2):
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    changes1 = [line[2:] for line in diff if line.startswith("- ")]
    changes2 = [line[2:] for line in diff if line.startswith("+ ")]
    return changes1, changes2



def process(data_chunk):
    ids = data_chunk['ids']
    code2content = data_chunk['code2content']
    results_data = data_chunk['results_data']
    previous_perf = data_chunk['previous_perf']

    train = data_chunk['train']
    code_bm25 = data_chunk['code_bm25']
    edit_opt_bm25 = data_chunk['edit_opt_bm25']
    edit_code_bm25 = data_chunk['edit_code_bm25']


    results = []
    differ = difflib.Differ()
    out_count=0
    for obj in tqdm(results_data):
        if obj['stop'] >= 1 or obj['idx'] not in ids:
            results.append(obj)
        elif obj['query'] not in previous_perf:
            obj['stop'] = 2
            results.append(obj)
            print(obj['idx'])
        else:
            perf = previous_perf[obj['query']]

            # update candidates
            obj['iteration_{}'.format(str(cfg.iteration))] = []
            for i in range(cfg.generation_number):
                if perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))] is None:
                    perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))] = 99999
                if perf['model_generated_potentially_faster_code_col_{}'.format(str(i))] not in code2content:
                    out_count+=1
                    code2content[perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]] = perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]
                obj['iteration_{}'.format(str(cfg.iteration))].append({'code':perf['model_generated_potentially_faster_code_col_{}'.format(str(i))], 'acc':perf['model_generated_potentially_faster_code_col_{}_acc'.format(str(i))], 'time':perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))], 'input_time':perf['input_time_mean'], 'content':code2content[perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]]})
            last_candidates = ''.join([i['code'] for i in obj['best_candidates'][:3]]) 
            obj['best_candidates'].extend(obj['iteration_{}'.format(str(cfg.iteration))])
            obj['best_candidates'] = selection(obj['best_candidates'], cfg)[:3]
            new_candidates = ''.join([i['code'] for i in obj['best_candidates'][:3]]) 


            # compare and select the best result
            if perf['model_generated_potentially_faster_code_col_time_mean'] is None:
                perf['model_generated_potentially_faster_code_col_time_mean'] = 99999
            obj['iteration_{}_results'.format(str(cfg.iteration))] = {'code':perf['model_generated_potentially_faster_code_col'], 'acc':perf['model_generated_potentially_faster_code_col_acc'], 
                                                                      'time':perf['model_generated_potentially_faster_code_col_time_mean'], 
                                                                      'input_time': perf['input_time_mean'], 'input_acc': perf['input_acc'],
                                                                      'reference_time': perf['reference_time_mean'], 'reference_acc': perf['reference_acc']}
            if better(obj['iteration_{}_results'.format(str(cfg.iteration))], obj['best_result']):
                obj['best_result'] = obj['iteration_{}_results'.format(str(cfg.iteration))]
            if last_candidates == new_candidates and obj['best_result']['acc'] == 1:
                obj['stop'] = 1
            if last_candidates == new_candidates:
                results.append(obj)
                continue

            if cfg.iteration<5:
                try:
                    query_abs = tokenize_code(abstract_code(obj['query']))
                except:
                    obj['pattern'] = []
                    results.append(obj)
                    continue
                code_scores = np.array(code_bm25.get_scores(query_abs))
                code_scores = (code_scores - code_scores.min()) / (code_scores.max() - code_scores.min())
                sim_scores = np.copy(code_scores)
                dissim_scores = np.copy(code_scores)
                for candidate in obj['best_candidates'][:cfg.beam_number]:
                    try:
                        candidate_abs = tokenize_code(abstract_code(candidate['code'])) 
                    except:
                        continue
                    edit_code_abs, edit_opt_abs = get_diff_lines(' '.join(query_abs).split('\n'), ' '.join(candidate_abs).split('\n'))
                    edit_code_abs = '\n'.join(edit_code_abs).split()
                    edit_opt_abs = '\n'.join(edit_opt_abs).split()
                    edit_scores = np.array(edit_code_bm25.get_scores(edit_code_abs))+np.array(edit_opt_bm25.get_scores(edit_opt_abs))
                    if edit_scores.max() - edit_scores.min()>0:
                        edit_scores = (edit_scores - edit_scores.min()) / (edit_scores.max() - edit_scores.min()) 
                        median_val = np.median(edit_scores)
                        sim_edit_scores = np.where(edit_scores < median_val, 0, edit_scores)
                        dissim_edit_scores = np.where(edit_scores > median_val, 0, np.max(edit_scores)-edit_scores)
                        sim_scores+=sim_edit_scores/cfg.beam_number
                        dissim_scores+=dissim_edit_scores/cfg.beam_number
                sim_top_indices  = np.argsort(-sim_scores)[0]
                sim_diff = list(differ.compare(train[sim_top_indices]['query'].split('\n'), train[sim_top_indices]['reference'].split('\n')))
                sim_pattern = [line for line in sim_diff if line.startswith("- ") or line.startswith("+ ")]
                dissim_top_indices  = np.argsort(-dissim_scores)[0]
                dissim_diff = list(differ.compare(train[dissim_top_indices]['query'].split('\n'), train[dissim_top_indices]['reference'].split('\n')))
                dissim_pattern = [line for line in dissim_diff if line.startswith("- ") or line.startswith("+ ")]
                obj['pattern'] = ['\n'.join(sim_pattern), '\n'.join(dissim_pattern)]
            
            results.append(obj)

    return results




def main(cfg):
    previous_perf = {}
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'test_execution_{}test.report'.format(cfg.model_name))) as f:
        for obj in f:
            previous_perf[remove_comments_and_docstrings(obj['code_v0_no_empty_lines'], cfg.lang)] = obj

    code2content = {}
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name))) as f:
        for obj in f:
            for code, content in zip(obj['prediction'], obj['detailed_prediction']):
                if isinstance(content, dict) and 'message' in content:
                    assert code in content['message']['content']
                    if cfg.iteration ==1:
                        code2content[code] = content['message']['content']
                    else:
                        code2content[code] = extract(code, content['message']['content'])
                else:
                    assert code in content
                    if cfg.iteration ==1:
                        code2content[code] = content
                    else:
                        code2content[code] = extract(code, content)


    train = []
    code_corpus = []
    edit_code_corpus = []
    edit_opt_corpus = []
    with jsonlines.open(cfg.training_data_path) as f:
        for obj in f:
            train.append(obj)
            if len(obj['query_abs'])>0 and len(obj['edit_code_abs'])>0 and len(obj['edit_opt_abs'])>0:
                code_corpus.append(obj['query_abs'])
                edit_code_corpus.append(obj['edit_code_abs'])
                edit_opt_corpus.append(obj['edit_opt_abs'])
    code_bm25 = BM25Okapi(code_corpus)
    edit_code_bm25 = BM25Okapi(edit_code_corpus)
    edit_opt_bm25 = BM25Okapi(edit_opt_corpus)


    ids = set()
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name))) as f:
        for obj in f:
            ids.add(obj['idx'])
            
    results_data = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'results.jsonl')) as f:
        for obj in f:
            results_data.append(obj)

    processes = int(40)
    pool = multiprocessing.Pool(processes)
    chunks_results_data = np.array_split(results_data, processes)
    chunks_data = []
    for i in range(processes):
        tmp_data={}
        tmp_data['cfg'] = cfg
        tmp_data['ids'] = ids
        tmp_data['code2content'] = code2content
        tmp_data['results_data'] = chunks_results_data[i]
        tmp_data['previous_perf'] = previous_perf
        tmp_data['train'] = train
        tmp_data['code_bm25'] = code_bm25
        tmp_data['edit_opt_bm25'] = edit_opt_bm25
        tmp_data['edit_code_bm25'] = edit_code_bm25
        chunks_data.append(tmp_data)
    chunk_results = pool.map(process, chunks_data)
    results = []
    for chunk in chunk_results:
        results.extend(chunk)
    assert len(results) == len(results_data)


    sp = 0
    ptr = 0
    faster_count = 0
    unique_count = 0
    input_time_sum = 0
    generated_test_sum = 0
    faster_input_time_sum = 0
    faster_generated_test_sum = 0
    unique_reference_time_sum = 0
    unique_generated_test_sum = 0
    for obj in results:
        acc = obj['best_result']['acc']
        input_time = obj['best_result']['input_time']
        generated_time = obj['best_result']['time']
        reference_time = obj['best_result']['reference_time']
        if acc==1 and generated_time<input_time:
            if generated_time<reference_time:
               unique_count+=1
               unique_reference_time_sum += reference_time
               unique_generated_test_sum += generated_time
            faster_count+=1
            faster_input_time_sum += input_time
            faster_generated_test_sum += generated_time
            input_time_sum += input_time
            generated_test_sum += generated_time
        else:
            input_time_sum += input_time
            generated_test_sum += input_time
    print("Current best results: ")
    print('OPT: ', faster_count)
    print('OPT(%): ', round(100*faster_count/len(results), 2))
    print('SP: ', round(faster_input_time_sum/faster_generated_test_sum, 2))
    print('PTR: ', round(input_time_sum/generated_test_sum, 2))
    
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, 'results.jsonl'), 'w') as f:
        f.write_all(results)

    if cfg.iteration == 5:
        final_results = []
        if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top1')):
            os.makedirs(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top1'))
        for i in results:
            final_results.append({'query': i['query'], 'prediction': [can['code'] for can in i['best_candidates'][:1]]})
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top1', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
            f.write_all(final_results)

        final_results = []
        if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top3')):
            os.makedirs(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top3'))
        for i in results:
            if len(i['best_candidates'])<3:
                i['best_candidates'] = (3-len(i['best_candidates']))*[i['best_candidates'][0]]+i['best_candidates']
            final_results.append({'query': i['query'], 'prediction': [can['code'] for can in i['best_candidates'][:3]]})
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top3', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
            f.write_all(final_results)

        final_results = []
        if not os.path.exists(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top5')):
            os.makedirs(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top5'))
        for i in results:
            if len(i['best_candidates'])<5:
                i['best_candidates'] = (5-len(i['best_candidates']))*[i['best_candidates'][0]]+i['best_candidates']
            final_results.append({'query': i['query'], 'prediction': [can['code'] for can in i['best_candidates'][:5]]})
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.mode, str(cfg.iteration), 'top5', 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
            f.write_all(final_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='', type=str)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--beam_number", default=3, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--generation_number", default=1, type=int)
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--generation_path", default=None, type=str, required=True)
    parser.add_argument("--training_data_path", default=None, type=str, required=True)
    cfg = parser.parse_args()
    main(cfg)


