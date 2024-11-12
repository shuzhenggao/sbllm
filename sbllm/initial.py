import re
import tokenize
import argparse
from io import StringIO
import jsonlines


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


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='chatgpt', type=str, required=True)
parser.add_argument("--lang", default='python', type=str, required=True)
cfg = parser.parse_args()


data = []
if cfg.model_name == 'gpt4':
    with jsonlines.open('../processed_data/{}/processed_test_gpt4.jsonl'.format(cfg.lang)) as f:
        for obj in f:
            data.append(obj)
else:
    with jsonlines.open('../processed_data/{}/processed_test.jsonl'.format(cfg.lang)) as f:
        for obj in f:
            data.append(obj)


previous_perf = {}
with jsonlines.open('../output/{}/cot/top5/test_execution_{}test.report'.format(cfg.lang, cfg.model_name)) as f:
    for obj in f:
        previous_perf[remove_comments_and_docstrings(obj['code_v0_no_empty_lines'],cfg.lang)] = obj

processed_data = []
for obj in data:
    best_result = {'code':'', 'acc':0, 'time':99999, 'input_time':previous_perf[obj['query']]['input_time_mean'], 'reference_time':previous_perf[obj['query']]['reference_time_mean']}
    processed_data.append({'idx':obj['idx'], 'query':obj['query'], 'reference':obj['reference'], 'stop':0, 'best_result':best_result, 'best_candidates':[], 'pattern':[]})
   
with jsonlines.open('../output/{}/initial_results_{}.jsonl'.format(cfg.lang, cfg.model_name), 'w') as f:
    f.write_all(processed_data)
