import os
import sys
import asyncio

from hadrian_vllm.main import process_element_id, process_element_ids


async def run_example():
    # send quarter first to make sure right
    answer, df = await process_element_id(
        text_prompt_path="data/prompts/prompt4.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/quarter_images/7_2_q3.png",
        element_id="D12",
        # model_name="gemini-2.0-flash-001",  # Out of Credits for gpt-4o for now
        model_name="gpt-4o",  # Out of Credits for gpt-4o for now
        n_shot_imgs=2,
        eg_per_img=3,
        examples_as_multiturn=False,
    )

    print(f"Answer for D12 Quarter : {answer}")
    print(df.columns)
    print(df.query('`Element ID` == "D12"'))

    answer, df = await process_element_id(
        text_prompt_path="data/prompts/prompt4.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg2.png",
        element_id="D12",
        # model_name="gemini-2.0-flash-001",
        model_name="gpt-4o",
        n_shot_imgs=2,
        eg_per_img=3,
        examples_as_multiturn=False,
    )

    print(f"Answer for D12: {answer}")
    print(df.columns)
    print(df.query('`Element ID` == "D12"'))

    # Try with multi-turn example format
    element_ids = ["T11", "D12", "D13", "T12", "D14", "T13", "D15", "T14"]
    answer_mt, df_mt = await process_element_ids(
        text_prompt_path="data/prompts/prompt4.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg2.png",
        element_ids=element_ids,
        # model_name="gemini-2.0-flash-001",
        model_name="gpt-4o",
        n_shot_imgs=5,
        eg_per_img=3,
        examples_as_multiturn=True,
    )

    print(f"""Answer for {element_ids} (multi-turn): {answer_mt}""")
    print(df_mt.columns)
    print(df_mt.query(f"`Element ID` in @element_ids"))


if __name__ == "__main__":
    asyncio.run(run_example())


# %%
# example_run.py
import asyncio
import os
import pandas as pd
import json
import argparse
from datetime import datetime

from hadrian_vllm.main import process_element_id, process_element_ids, run_evaluation
from hadrian_vllm.utils import load_csv, extract_assembly_and_page_from_filename
from hadrian_vllm.evaluation import evaluate_answer, calculate_metrics
from hadrian_vllm.prompt_generator import extract_element_ids_from_image


async def run_single_example():
    """Run a single example with one element ID"""
    print("\n=== Running Single Element ID Example ===")

    answer, df = await process_element_id(
        text_prompt_path="data/prompts/3_single_elem_asem6_few_shot.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png",
        element_id="D12",
        model_name="gpt-4o",
        n_shot_imgs=3,
        eg_per_img=3,
        examples_as_multiturn=False,
    )

    print(f"Answer for D12: {answer}")

    # Validate against ground truth
    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    df = load_csv(csv_path)
    ground_truth = df[df["Element ID"] == "D12"]["Specification"].iloc[0]

    print(f"Ground truth: {ground_truth}")

    try:
        is_correct = evaluate_answer(answer, ground_truth)
        print(f"Correct: {is_correct}")
    except Exception as e:
        print(f"Validation error: {e}")


async def run_multi_element_example():
    """Run an example with multiple element IDs"""
    print("\n=== Running Multiple Element IDs Example ===")

    # Select an image and get some element IDs
    image_path = "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png"
    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"

    # Get element IDs for this image
    element_ids = extract_element_ids_from_image(csv_path, image_path)[:3]  # Get first 3 elements

    print(f"Using element IDs: {element_ids}")

    # Process multiple element IDs
    answers, df = await process_element_ids(
        text_prompt_path="data/prompts/3_single_elem_asem6_few_shot.txt",
        csv_path=csv_path,
        eval_dir="data/eval_on/single_images/",
        question_image=image_path,
        element_ids=element_ids,
        model_name="gpt-4o",
        n_shot=3,
        eg_per_img=1,  # 1 example per image, with multiple elements
        examples_as_multiturn=False,
    )

    print(f"Answers: {answers}")

    # Validate against ground truth
    ground_truths = {}
    for element_id in element_ids:
        ground_truth = df[df["Element ID"] == element_id]["Specification"].iloc[0]
        ground_truths[element_id] = ground_truth

    print("\nValidation Results:")
    for i, element_id in enumerate(element_ids):
        ground_truth = ground_truths[element_id]
        answer = answers[i]

        print(f"\nElement ID: {element_id}")
        print(f"Answer: {answer}")
        print(f"Ground truth: {ground_truth}")

        try:
            is_correct = evaluate_answer(answer, ground_truth)
            print(f"Correct: {is_correct}")
        except Exception as e:
            print(f"Validation error: {e}")


async def run_multiturn_example():
    """Run an example with multi-turn conversation"""
    print("\n=== Running Multi-Turn Example ===")

    # Select an image and get some element IDs
    image_path = "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png"
    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    element_id = "D12"

    # Process with multi-turn conversation
    answer, df = await process_element_id(
        text_prompt_path="data/prompts/3_single_elem_asem6_few_shot.txt",
        csv_path=csv_path,
        eval_dir="data/eval_on/single_images/",
        question_image=image_path,
        element_id=element_id,
        model_name="gpt-4o",
        n_shot=3,
        eg_per_img=3,
        examples_as_multiturn=True,
    )

    print(f"Answer for {element_id}: {answer}")

    # Validate against ground truth
    ground_truth = df[df["Element ID"] == element_id]["Specification"].iloc[0]

    print(f"Ground truth: {ground_truth}")

    try:
        is_correct = evaluate_answer(answer, ground_truth)
        print(f"Correct: {is_correct}")
    except Exception as e:
        print(f"Validation error: {e}")


async def run_full_evaluation(model_name="gpt-4o", multiturn=False, num_images=2):
    """Run a small-scale full evaluation"""
    print(f"\n=== Running Full Evaluation (model: {model_name}, multiturn: {multiturn}) ===")

    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    eval_dir = "data/eval_on/single_images/"

    # Get a limited number of images for the example
    image_files = [f for f in os.listdir(eval_dir) if f.endswith(".png")][:num_images]
    question_images = [os.path.join(eval_dir, img) for img in image_files]

    # Group element IDs by image
    element_ids_by_image = {}
    for img_path in question_images:
        # Get element IDs for this image (limited to 3 for the example)
        element_ids = extract_element_ids_from_image(csv_path, img_path)[:3]
        if element_ids:
            element_ids_by_image[img_path] = element_ids

    # Run evaluation
    results = await run_evaluation(
        text_prompt_path="data/prompts/3_single_elem_asem6_few_shot.txt",
        csv_path=csv_path,
        eval_dir=eval_dir,
        question_images=list(element_ids_by_image.keys()),
        element_ids_by_image=element_ids_by_image,
        model_names=[model_name],
        n_shot=3,
        eg_per_img=1,
        num_completions=1,
        examples_as_multiturn=multiturn,
    )

    # Print overall results
    print("\nOverall Results:")
    for model, metrics in results.items():
        if "error" in metrics:
            print(f"\n{model}: Error - {metrics['error']}")
        else:
            print(f"\n{model}:")
            print(f"Correct: {metrics['percent_correct']:.2f}%")
            print(f"Incorrect: {metrics['percent_incorrect']:.2f}%")
            print(f"Unanswered: {metrics['percent_unanswered']:.2f}%")
            print(f"Accuracy: {metrics['accuracy']:.4f}")


async def main2():
    parser = argparse.ArgumentParser(description="Run examples of GD&T extraction")
    parser.add_argument(
        "--example",
        type=str,
        choices=["single", "multi", "multiturn", "full"],
        default="single",
        help="Type of example to run",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("data/prompt_config", exist_ok=True)
    os.makedirs("data/llm_completions", exist_ok=True)

    # Run the selected example
    if args.example == "single":
        await run_single_example()
    elif args.example == "multi":
        await run_multi_element_example()
    elif args.example == "multiturn":
        await run_multiturn_example()
    elif args.example == "full":
        await run_full_evaluation(args.model)
    else:
        print("Unknown example type. Please choose one of: single, multi, multiturn, full")


# if __name__ == "__main__":
#    asyncio.run(main())
