lang=python
public_test_case_path=../processed_data/public_test_cases # for generation
private_test_case_path=../processed_data/generated_test_cases # for final testing
training_data_path=../processed_data/$lang/processed_train.jsonl
model_name=chatgpt
generation_number=4
beam_number=3
baseline_data_path=../processed_data/$lang/processed_test.jsonl
output_path=../output/$lang
base_mode=sbllm

generation_path=../output
api_idx=0
restart_pos=0

if [ ! -d "../output/$lang/$base_mode/0" ]; then
  mkdir -p ../output/$lang/$base_mode/1
  cp ../output/$lang/initial_results_$model_name.jsonl ../output/$lang/$base_mode/results.jsonl
  cp -r ../output/$lang/cot ../output/$lang/$base_mode
  mv ../output/$lang/$base_mode/cot ../output/$lang/$base_mode/0
fi


for i in {0..4} 
do
  output_path=../output
  mode=$base_mode
  python evol_query.py  --from_file --mode $mode --generation_path $generation_path --model_name $model_name --baseline_data_path $baseline_data_path \
                        --api_idx $api_idx --restart_pos $restart_pos --generation_number $generation_number --iteration $i --beam_number $beam_number --lang $lang 

  if [ $? -eq 0 ]; then
    output_path=../output/$lang/$base_mode
    mode="${i}"
    python evaluate.py  --do_train --mode $mode --lang $lang --process_number 60 \
                        --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $public_test_case_path
    output_path=../output/$lang
    mode=$base_mode
    python merge.py  --iteration $i --generation_number $generation_number  --lang $lang --model_name $model_name --mode $mode --training_data_path $training_data_path --generation_path $generation_path --beam_number $beam_number
  else
    echo "evaluate.py $i failed with exit code $?."
    break
  fi
done

output_path=../output/$lang/$base_mode
mode=top1
python evaluate.py  --do_train --mode $mode --lang $lang  --process_number 60 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path

output_path=../output/$lang/$base_mode
mode=top3
python evaluate.py  --do_train --mode $mode --lang $lang  --process_number 60 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path

output_path=../output/$lang/$base_mode
mode=top5
python evaluate.py  --do_train --mode $mode --lang $lang  --process_number 60 \
                    --output_path $output_path --model_name $model_name  --slice 1 --testing_number 0 --test_case_path $private_test_case_path

