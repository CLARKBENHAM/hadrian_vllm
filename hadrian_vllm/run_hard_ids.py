# %%
# Iter fast on Run the problems everythings screwed up on
import sys
import os
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
import logging
import traceback
import nest_asyncio

nest_asyncio.apply()

from hadrian_vllm.prompt_generator import element_ids_per_img_few_shot
from hadrian_vllm.model_caller import call_model, preload_images
from hadrian_vllm.result_processor import extract_answer, save_results
from hadrian_vllm.evaluation import calculate_metrics, evaluate_answer
from hadrian_vllm.main import process_element_ids
from hadrian_vllm.utils import (
    load_csv,
    save_df_with_results,
    get_git_hash,
    get_current_datetime,
    get_element_details,
    extract_assembly_and_page_from_filename,
    is_debug_mode,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# import logging
#
# def set_all_loggers_level(level=logging.WARNING):
#     # Set the root logger level
#     logging.getLogger().setLevel(level)
#     # Iterate over all loggers that have been created and set their level
#     for logger_name, logger_obj in logging.root.manager.loggerDict.items():
#         if isinstance(logger_obj, logging.Logger):
#             logger_obj.setLevel(level)
#
#
# set_all_loggers_level(logging.WARNING)


async def run_hard_qs(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_images,
    element_ids_by_image,
    model_names,
    n_shot_imgs=21,
    eg_per_img=50,
    n_element_ids=1,
    num_completions=1,
    examples_as_multiturn=False,
):
    """
    Run evaluation for multiple images and element IDs.

    Args:
        text_prompt_path: Path to the prompt template
        csv_path: Path to the GD&T data CSV
        eval_dir: Directory with evaluation images
        question_images: List of question image paths
        element_ids_by_image: Dictionary mapping image paths to lists of element IDs
        model_names: List of model names to evaluate
        n_shot_imgs: Number of few-shot examples
        eg_per_img: Number of examples per image
        n_element_ids: Max number of element ids to send per request
        num_completions: Number of completions to generate for each query
        examples_as_multiturn: Whether to format examples as multiple turns

    Returns:
        Dictionary of metrics by model
    """
    if num_completions > 1:
        assert False, "Doesn't combine in smart way, or use saving code"
    # Load the DataFrame
    df = load_csv(csv_path)

    # Results by model
    results_by_model = {}
    model_dfs = {model_name: df.copy() for model_name in model_names}
    latest_result_columns = {model_name: None for model_name in model_names}

    # Create all tasks first
    all_tasks = []  # List to track (model_name, img_path, element_ids, task)
    preload_images(
        question_images
        + [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".png")]
    )

    # for IO, but sending images with request so might delay a tad
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=12)
    )  # adjust here, seems litellm fails with 64 threads. Debug failed with 16 also?

    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        for img_path in question_images:
            all_element_ids = element_ids_by_image.get(img_path, [])
            if not all_element_ids:
                continue

            print(f"Processing image: {os.path.basename(img_path)}")
            print(f"Element IDs: {all_element_ids}")

            for start in range(0, len(all_element_ids), n_element_ids):
                element_ids = all_element_ids[start : start + n_element_ids]
                print(f"Batch Element IDs: {element_ids}")

                for i in range(num_completions):
                    task = process_element_ids(
                        text_prompt_path,
                        csv_path,
                        eval_dir,  # have to change to not use part of image
                        img_path,
                        element_ids,
                        model_name,
                        n_shot_imgs,
                        eg_per_img,
                        examples_as_multiturn,
                        cache=num_completions == 1,
                    )
                    all_tasks.append((model_name, img_path, element_ids, i, task))
                    # with asyncio.sleep(0) tasks would still take 0.5 sec to start but still getting a few gemini rate limits
                    # not sure what's happenign there.
                    # await asyncio.sleep(0.1)  # so so many tasks not all started right away
        # await asyncio.sleep(0.5)  # only start all req given image at same time
        # Try and create all tasks as fast as possible, but run them concurrenlty later
    # Run all tasks concurrently
    all_results = await asyncio.gather(*(task for _, _, _, _, task in all_tasks))

    # Process results
    total_right = 0
    for (model_name, img_path, element_ids, _, _), (preds, _model_df) in zip(
        all_tasks, all_results
    ):
        answer_col = _model_df.columns[-1]
        if latest_result_columns[model_name] is None:
            latest_result_columns[model_name] = answer_col
            model_dfs[model_name][answer_col] = None
        assert (
            latest_result_columns[model_name] == answer_col
        ), f"Same run_eval had different col {latest_result_columns[model_name]} {answer_col}"

        assembly_id, page_id = extract_assembly_and_page_from_filename(img_path)
        print(assembly_id, page_id)

        print(
            f"\nResults for model {model_name}, image {os.path.basename(img_path)}, elements"
            f" {element_ids}:"
        )
        for answer, element_id in zip(preds, element_ids):
            # store real
            row = (
                (model_dfs[model_name]["Assembly ID"] == assembly_id)
                & (model_dfs[model_name]["Page ID"] == page_id)
                & (model_dfs[model_name]["Element ID"] == element_id)
            )
            model_dfs[model_name].loc[row, latest_result_columns[model_name]] = answer
            # print(model_dfs[model_name].loc[row, latest_result_columns[model_name]], answer)
            # save multiple completions here?

            element_details = get_element_details(
                csv_path,
                element_id,
                assembly_id=assembly_id,
                page_id=page_id,
            )
            real_answer = element_details["Specification"]
            print(
                f"{evaluate_answer(answer, real_answer)} Real: `{real_answer}`, Model: `{answer}`"
            )
            total_right += evaluate_answer(answer, real_answer)

        # If we need multiple completions, get them and use the best one
        if num_completions > 1:
            assert False, "Doesn't combine in smart way, or use saving code"
    print(f"{total_right}/{len(all_results)}")

    # Calculate metrics and save results for each model
    for model_name in model_names:
        try:
            # Calculate metrics
            model_df = model_dfs[model_name]
            latest_result_column = latest_result_columns[model_name]
            print(latest_result_column)
            metrics = calculate_metrics(model_df, latest_result_column)
            results_by_model[model_name] = metrics

            # Don't Save the DataFrame here
            # output_path = f"data/results/{latest_result_column.replace(' ', '_')}.csv"
            # save_df_with_results(model_df, latest_result_column, output_path)

            # Print metrics
            print(f"\nResults for {model_name}:")
            print(f"Correct: {metrics['correct']} ({metrics['percent_correct']:.2f}%)")
            print(f"Incorrect: {metrics['incorrect']} ({metrics['percent_incorrect']:.2f}%)")
            print(f"Unanswered: {metrics['unanswered']} ({metrics['percent_unanswered']:.2f}%)")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            results_by_model[model_name] = {"error": str(e)}

    return results_by_model


async def main():
    parser = argparse.ArgumentParser(description="Extract GD&T data from CAD renders")
    parser.add_argument(
        "--prompt",
        type=str,
        default="data/prompts/prompt4.txt",
        help="Path to the prompt template",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        help="Path to the GD&T data CSV",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="data/eval_on/single_images/",
        help="Directory with evaluation images",
    )
    parser.add_argument(
        "--question_image",
        type=str,
        default=None,  # "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png",
        help="Path to specific image we're running here to test",
    )
    # parser.add_argument(
    #     "--og_image",
    #     type=str,
    #     default=None,  # "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png",
    #     help="Path to original image the clip came from",
    # )
    parser.add_argument(
        "--hard_element_ids",
        type=str,
        nargs="+",
        default=[
            "D11-2",
            "D13",
            "D9-1",
            "D9-2",
            "DF6",
            "T11",
            "T12",
            "T13",
            "T14",
            "T15",
        ],
        help="Element ID to query; see hard_element_ids_by_image",
    )
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-001", help="Model to use")
    parser.add_argument("--n_shot_imgs", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--eg_per_img", type=int, default=3, help="Number of examples per image")
    parser.add_argument(
        "--n_element_ids",
        type=int,
        default=1,
        help=(
            "Maximum number of element ids to send at once to process_element_ids for a given image"
        ),
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=1,
        help="Number of completions to generate for each query",
    )
    parser.add_argument(
        "--multiturn", action="store_true", help="Format examples as multiple turns"
    )
    parser.add_argument(
        "--eval_all", action="store_true", help="Run evaluation on all element IDs in the CSV"
    )
    # only in evalutions.py
    parser.add_argument("--eval-easy", action="store_true", help="Use easy evaluation mode")

    args = parser.parse_args()

    # Process a single element ID
    df = load_csv(args.csv)

    # Group element IDs by image (image: [element_ids])
    # hard without answers in prompt
    hard_element_ids_by_image = {
        "data/eval_on/single_images/nist_ftc_11_asme1_rb_elem_ids_pg1.png": [
            "T2",
            "D1",
            "D3",
            "T3",
        ],
        "data/eval_on/single_images/nist_ftc_06_asme1_rd_elem_ids_pg1.png": [
            "D11-1",
            "D11-2",
            "D12-1",
            "D13",
            "D2",
            "D8",
            "D9-1",
            "D9-2",
            "DF6",
            "T11",
            "T12",
            "T13",
            "T14",
            "T15",
            "T5",
            "T8",
        ],
    }

    # Hard with answers in prompt
    # python scripts/aggr_multiple_evaluations.py  --threshold 0.8 --eval-easy  --hours 90 --print_difficult  | ggrep -Po 'Assembly: \d+, Page: 1, Element: .*' d | sort
    hard_element_ids_by_image = {
        "data/eval_on/single_images/nist_ftc_11_asme1_rb_elem_ids_pg1.png": [
            "T2",
            "D1",
        ],
        "data/eval_on/single_images/nist_ftc_06_asme1_rd_elem_ids_pg1.png": [
            "D11-2",
            "D13",
            "D9-1",
            "D9-2",
            "DF6",
            "T11",
            "T12",
            "T13",
            "T14",
            "T15",
        ],
    }
    hard_element_ids_for_clip = {args.question_image: args.hard_element_ids}

    # assert set(args.hard_element_ids) <= set(
    #     hard_element_ids_by_image[
    #         "data/eval_on/single_images/nist_ftc_06_asme1_rd_elem_ids_pg1.png"
    #     ]
    # ), f"probably wrong element id in {args.hard_element_ids}"

    # Run evaluation
    model_names = [args.model]  # You can add more models here if needed
    results = await run_hard_qs(
        # args.prompt,
        "data/prompts/prompt7.txt",
        args.csv,
        args.eval_dir,
        [args.question_image],
        hard_element_ids_for_clip,
        model_names,
        args.n_shot_imgs,
        args.eg_per_img,
        args.n_element_ids,
        args.num_completions,
        args.multiturn,
    )

    print("\nOverall Results:")
    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"\n{model_name}: Error - {metrics['error']}")
        else:
            print(f"\n{model_name}:")
            print(f"Correct: {metrics['percent_correct']:.2f}%")
            print(f"Incorrect: {metrics['percent_incorrect']:.2f}%")
            print(f"Unanswered: {metrics['percent_unanswered']:.2f}%")


if __name__ == "__main__":
    # asyncio.run(main())

    # Try and run directly, but warn this overwrites the same image from multiple sources
    element_ids_by_image = {
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_small.jpg": [
            "D10",
            "T13",
            "D11-1",
            "T14",
            "D11-2",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_large.jpg": [
            "STR5",
            "T23",
            "D14",
            "D10",
            "T13",
            "D11-1",
            "D11-2",
            "T14",
            "D8",
            "D9-1",
            "D9-2",
            "T12",
            "T1",
            "DF1",
            "CS1",
            "CS2",
            "Z1",
            "A",
            "B",
            "C",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D8_small.jpg": [
            "D8",
            "D9-1",
            "D9-2",
            "T12",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_med.jpg": [
            "D15",
            "D13",
            "D7",
            "T11",
            "T4",
            "T5",
            "DF4",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D12_small.jpg": [
            "CS7",
            "D12-1",
            "T15",
            "STR4",
            "D12-2",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_medium.png": [
            "D8",
            "D9-1",
            "D9-2",
            "D10",
            "D11-1",
            "D11-2",
            "T12",
            "T13",
            "T14",
        ],
        "data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_large.jpg": [
            "D7",
            "D12-1",
            "D12-2",
            "D13",
            "D15",
            "T4",
            "T5",
            "T11",
            "T15",
            "DF4",
            "DF6",
            "CS3",
            "CS5",
            "CS6",
            "CS7",
            "STR1",
            "STR4",
        ],
    }
    element_ids_by_image = {
        k: [
            i
            for i in v
            if i
            in [
                "D11-1",
                "D11-2",
                "D12-1",
                "D13",
                "D2",
                "D8",
                "D9-1",
                "D9-2",
                "DF6",
                "T11",
                "T12",
                "T13",
                "T14",
                "T15",
                "T5",
                "T8",
            ]
        ]
        for k, v in element_ids_by_image.items()
    }
    text_prompt_path = "data/prompts/prompt7.txt"

    loop = asyncio.get_running_loop()
    task = loop.create_task(
        run_hard_qs(
            text_prompt_path,
            "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
            "data/eval_on/single_images/",
            list(element_ids_by_image.keys()),
            element_ids_by_image,
            ["gemini-2.0-flash-001"],
            n_shot_imgs=21,
            eg_per_img=50,
            n_element_ids=1,
            num_completions=1,
            examples_as_multiturn=True,
        )
    )
    result = loop.run_until_complete(task)
    print(result)


def f():
    """
    for f in `ls data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1/`; do
    echo "python -u hadrian_vllm/main.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1/$f --hard_element_ids    --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1"
    done

    python scripts/elem_ids_per_img.py  to get:
    element_ids_by_image={'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_small.jpg': ['D10', 'T13', 'D11-1', 'T14', 'D11-2'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_large.jpg': ['STR5', 'T23', 'D14', 'D10', 'T13', 'D11-1', 'D11-2', 'T14', 'D8', 'D9-1', 'D9-2', 'T12', 'T1', 'DF1', 'CS1', 'CS2', 'Z1', 'A', 'B', 'C'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D8_small.jpg': ['D8', 'D9-1', 'D9-2', 'T12'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_med.jpg': ['D15', 'D13', 'D7', 'T11', 'T4', 'T5', 'DF4'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D12_small.jpg': ['CS7', 'D12-1', 'T15', 'STR4', 'D12-2'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_medium.png': ['D8', 'D9-1', 'D9-2', 'D10', 'D11-1', 'D11-2', 'T12', 'T13', 'T14'], 'data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_large.jpg': ['D7', 'D12-1', 'D12-2', 'D13', 'D15', 'T4', 'T5', 'T11', 'T15', 'DF4', 'DF6', 'CS3', 'CS5', 'CS6', 'CS7', 'STR1', 'STR4']}
    # no hallucinations on hard ids
    for file, elem in sorted(d.items()):
        elem =   [i for i in elem if i in ["D11-1","D11-2","D12-1","D13","D2","D8","D9-1","D9-2","DF6","T11","T12","T13","T14","T15","T5","T8"]]
        elem  = " ".join([f"'{i}'" for i in elem])
        print(f"python -u hadrian_vllm/run_hard_ids.py --question_image {file} --hard_element_ids {elem}  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt")


    D10_medium ['D10', 'D11-1', 'D11-2', 'D8', 'D9-1', 'D9-2', 'T12', 'T13', 'T14'] D10 T13 D11-2 D11-1 T14 D8 D9-1 D9-2 T12



    PROMPT=data/prompts/prompt4.txt
    MODEL=gemini-2.0-flash-001

    # Hard ids only
    echo "" > data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_large.jpg --hard_element_ids  'T13' 'D11-1' 'D11-2' 'T14' 'D8' 'D9-1' 'D9-2' 'T12'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_medium.png jpg --hard_element_ids  'D8' 'D9-1' 'D9-2' 'D11-1' 'D11-2' 'T12' 'T13' 'T14'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_small.jpg --hard_element_ids  'T13' 'D11-1' 'T14' 'D11-2'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D12_small.jpg --hard_element_ids  'D12-1' 'T15'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_large.jpg --hard_element_ids  'D12-1' 'D13' 'T5' 'T11' 'T15' 'DF6'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_med.jpg --hard_element_ids  'D13' 'T11' 'T5'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D8_small.jpg --hard_element_ids  'D8' 'D9-1' 'D9-2' 'T12'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1 | tee -a data/printout/hard.txt
    cat data/printout/hard.txt

    # all ids on image
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_small.jpg --hard_element_ids  'D10' 'T13' 'D11-1' 'T14' 'D11-2'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_large.jpg --hard_element_ids  'STR5' 'T23' 'D14' 'D10' 'T13' 'D11-1' 'D11-2' 'T14' 'D8' 'D9-1' 'D9-2' 'T12' 'T1' 'DF1' 'CS1' 'CS2' 'Z1' 'A' 'B' 'C'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D8_small.jpg --hard_element_ids  'D8' 'D9-1' 'D9-2' 'T12'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_med.jpg --hard_element_ids  'D15' 'D13' 'D7' 'T11' 'T4' 'T5' 'DF4'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D12_small.jpg --hard_element_ids  'CS7' 'D12-1' 'T15' 'STR4' 'D12-2'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D10_medium.png jpg --hard_element_ids  'D8' 'D9-1' 'D9-2' 'D10' 'D11-1' 'D11-2' 'T12' 'T13' 'T14'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1
    python -u hadrian_vllm/run_hard_ids.py --question_image data/eval_on/clipped_nist_ftc_06_asme1_rd_elem_ids_pg1//D15_large.jpg --hard_element_ids  'D7' 'D12-1' 'D12-2' 'D13' 'D15' 'T4' 'T5' 'T11' 'T15' 'DF4' 'DF6' 'CS3' 'CS5' 'CS6' 'CS7' 'STR1' 'STR4'  --model $MODEL --prompt $PROMPT  --csv 'data/fsi_labels/Hadrian Vllm test case - Final Merge.csv' --eval_dir data/eval_on/single_images/ --n_shot_imgs 21 --eg_per_img 50 --n_element_ids 1 --num_completions 1 --multiturn 2>&1

    """
    pass
