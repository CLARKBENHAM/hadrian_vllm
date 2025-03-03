# Claude prompt re-write 5 img each 50
python -u hadrian_vllm/main.py --prompt data/prompts/prompt6_claude_try.txt --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model gemini-2.0-flash-001 --n_shot_imgs 5 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all 2>&1 | tee data/printout/assem_6_11_n5_eg50_prompt6_claude_try.txt
python -u hadrian_vllm/main.py --prompt data/prompts/prompt6_claude_try.txt --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model gemini-2.0-flash-001 --n_shot_imgs 5 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all --eval-easy 2>&1 | tee data/printout/assem_6_11_n5_eg50_prompt6_claude_try_easy.txt

# O3
python -u hadrian_vllm/main.py --prompt data/prompts/prompt6_o3_try.txt --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model gemini-2.0-flash-001 --n_shot_imgs 5 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all 2>&1 | tee data/printout/assem_6_11_n5_eg50_prompt6_o3_try.txt
python -u hadrian_vllm/main.py --prompt data/prompts/prompt6_o3_try.txt --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/assem_6_11_single_images/ --model gemini-2.0-flash-001 --n_shot_imgs 5 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn --eval_all --eval-easy 2>&1 | tee data/printout/assem_6_11_n5_eg50_prompt6_o3_try_easy.txt
