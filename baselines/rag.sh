model_name=chatgpt
lang=python
generation_path=../output/$lang
training_data_path=../processed_data/$lang/processed_train.jsonl
valid_data_path=../processed_data/$lang/processed_val.jsonl
test_data_path=../processed_data/$lang/processed_test.jsonl
private_test_case_path=../processed_data/generated_test_cases


mode=rag
baseline_data_path=../processed_data/$lang/rag.jsonl
python baselines.py  --lang $lang  --mode $mode --generation_path $generation_path --model_name $model_name --baseline_data_path $baseline_data_path --generation_number 5



base_mode=rag
output_path=../output/$lang/$base_mode
mode=top1
python evaluate.py  --do_train --mode $mode --lang $lang --process_number 10 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path


output_path=../output/$lang/$base_mode
mode=top3
python evaluate.py  --do_train --mode $mode --lang $lang --process_number 10 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path



output_path=../output/$lang/$base_mode
mode=top5
python evaluate.py  --do_train --mode $mode --lang $lang --process_number 10 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path

