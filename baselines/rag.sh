model_name=chatgpt
lang=cpp
generation_path=../output/$lang
training_data_path=../processed_data/$lang/processed_train.jsonl
valid_data_path=../processed_data/$lang/processed_val.jsonl
test_data_path=../processed_data/$lang/processed_test.jsonl
public_test_case_path=../processed_data/public_test_cases


mode=rag
baseline_data_path=../processed_data/$lang/cpp_rag.jsonl
python3 baselines.py  --lang $lang  --mode $mode --generation_path $generation_path --model_name $model_name --baseline_data_path $baseline_data_path --generation_number 5

