# hadrian_vllm
Take CAD Model of ideal shape, Blueprint with tolerances, return unified standard.

``` # python-occ only on conda
conda activate
```


from OCC.Core.STEPCAFControl import STEPCAFControl_Reader

conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba


To run vertex ai, needed for gemini-2.0 (Not sure if True)
```
gcloud auth application-default login
```


To use this system, you can:

1. For a single query:
```bash
python -m gdt_extraction.main --prompt data/prompts/prompt4.txt --csv "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv" --eval_dir data/eval_on/single_images/ --image data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png --element_id D12 --model gpt-4o
```

2. To use multi-turn examples:
```bash
python -m gdt_extraction.main --prompt data/prompts/prompt4.txt --csv "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv" --eval_dir data/eval_on/single_images/ --image data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png --element_id D12 --model gpt-4o --multiturn
```

3. For batch evaluation:
```bash
python -m gdt_extraction.main --prompt data/prompts/prompt4.txt --csv "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv" --eval_dir data/eval_on/single_images/ --model gpt-4o --num_completions 1 --eval_all
```

```
python hadrian_vllm/main.py --prompt data/prompts/prompt4.txt --csv data/fsi_labels/HadrianVllmtestcase-FinalMerge.csv --eval_dir data/eval_on/single_images/ --model gemini-2.0-flash-001 //o1 --n_shot_imgs 2 --eg_per_img 4 --n_element_ids 1 --num_completions 1 --eval_all
```

The system efficiently extracts GD&T/PMI information from renders, with proper caching, error handling, and evaluation metrics.



# Order of Files - Chronological Work Order
## Making basic test case from NIST Data
make_basic_test_cases.py and stack_xlsx_files.py, were used to generated csvs which when fixed w/ llm help in https://docs.google.com/spreadsheets/d/1tdtUZaJAdkri25PN4LroGV-EJTjW0Uh98TclfjHdJlk/edit?gid=0#gid=0
created the test data answer keys at `data/fsi_labels/Hadrian Vllm test case - Final Merge.csv`. Gemini 2.0 pro was about the same as O1 in terms of transcribing the special GD&T unicode chars; I had added GD&T chars as part of the prompt. I removed the STR's that were for transcribing notes off the pages from final answer, only want to measure element IDs.

Replaced emoji m with text M.
Removed entries for transcriping full note text on diagram.
deleted rows with `VSI1`, were circles with B,C,D in big letters but not GD&T/PMI data. Not sure what for

### Extracting element IDs
`scripts/elem_ids_per_img.py` was used for first pass, then manually checked against few-shot Gemini-2.0-pro chat.
`scripts/elem_ids_per_img_quarters.py` splits image into quarters and gets ids from each quarter. Much more accurate and should be used instead

### Prompts
from prompt4 on we're writing a prompt we expect to replace text into `{{{Example}}}` and `{{{Question}}}`. If multi-turn it's assumed the system prompt is everything before `{{{Example}}}`

