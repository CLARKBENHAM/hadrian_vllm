# Try giving each few shot had answers; using gpt-4o for comparision
PROMPT=data/prompts/prompt6_o3_v2_few_shot_ans.txt
for MODEL in gemini-2.0-flash-001 gpt-4o; do
	python -u hadrian_vllm/main.py --prompt $PROMPT --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model $MODEL --n_shot_imgs 3 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all 2>&1 | tee -a data/printout/assem_6_11_prompt6_o3_v2_few_shot_ans_${MODEL}_n3_eg50.txt
	python -u hadrian_vllm/main.py --prompt $PROMPT --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model $MODEL --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all 2>&1 | tee -a  data/printout/assem_6_11_prompt6_o3_v2_few_shot_ans_${MODEL}_n21_eg50.txt
done

# print easy evals
for MODEL in gemini-2.0-flash-001 gpt-4o; do
	echo $MODEL
	python -u hadrian_vllm/main.py --prompt $PROMPT --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model $MODEL --n_shot_imgs 3 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all --eval-easy 2>&1 | tail -13
	python -u hadrian_vllm/main.py --prompt $PROMPT --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model $MODEL --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all --eval-easy 2>&1 | tail -13
done
