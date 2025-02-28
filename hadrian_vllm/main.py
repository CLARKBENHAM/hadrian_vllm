# main.py
import os
import argparse
import asyncio
import pandas as pd
import json

from hadrian_vllm.prompt_generator import element_ids_per_img_few_shot
from hadrian_vllm.model_caller import call_model, get_openai_messages
from hadrian_vllm.result_processor import extract_answer, save_results
from hadrian_vllm.evaluation import calculate_metrics, evaluate_answer
from hadrian_vllm.utils import (
    load_csv,
    save_df_with_results,
    get_git_hash,
    get_current_datetime,
    get_element_details,
    extract_assembly_and_page_from_filename,
)


async def process_element_id(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_image,
    element_id,
    model_name,
    n_shot_imgs=3,
    eg_per_img=3,
    examples_as_multiturn=False,
    cache=True,
):
    """
    Process a single element ID query.

    Args:
        text_prompt_path: Path to the prompt template
        csv_path: Path to the GD&T data CSV
        eval_dir: Directory with evaluation images
        question_image: Path to the question image
        element_id: Element ID to query
        model_name: Name of the model to use
        n_shot_imgs: Number of few-shot example images
        eg_per_img: Number of examples per image
            total example ids: n_shot_imgs*eg_per_img*1
        examples_as_multiturn: Whether to format examples as multiple turns
        cache should it be used
    Returns:
        The extracted answer and updated DataFrame
    """
    # Generate the prompt and get image paths
    assert isinstance(element_id, str), element_id

    prompt_or_messages, image_paths, config = element_ids_per_img_few_shot(
        text_prompt_path,
        csv_path,
        eval_dir,
        question_image,
        [element_id],
        n_shot_imgs,
        eg_per_img,
        examples_as_multiturn,
    )

    # Call the model
    if examples_as_multiturn:
        response = await call_model(prompt_or_messages, model=model_name, cache=cache)
    else:
        response = await call_model(prompt_or_messages, image_paths, model_name, cache=cache)

    # Extract the answer
    answer = extract_answer(response, element_id)

    # Load the DataFrame
    df = load_csv(csv_path)

    # Update config with model info
    config["model_name"] = model_name

    # Save the results
    df = save_results(df, prompt_or_messages, image_paths, element_id, response, answer, config)

    return answer, df


async def process_element_ids(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_image,
    element_ids,
    model_name,
    n_shot_imgs=3,
    eg_per_img=3,
    examples_as_multiturn=False,
    cache=True,
):
    """
    Process multiple element IDs in a batch.

    Args:
        text_prompt_path: Path to the prompt template
        csv_path: Path to the GD&T data CSV
        eval_dir: Directory with evaluation images
        question_image: Path to the question image
        element_ids: List of element IDs to query
        model_name: Name of the model to use
        n_shot_imgs: Number of few-shot images
        eg_per_img: Number of examples per image
            total example ids: n_shot_imgs*eg_per_img*len(element_ids)
        examples_as_multiturn: Whether to format examples as multiple turns
        cache: should the cache be used or call regenerated
    Returns:
        List of extracted answers and updated DataFrame
    """
    # Generate the prompt and get image paths
    prompt_or_messages, image_paths, config = element_ids_per_img_few_shot(
        text_prompt_path,
        csv_path,
        eval_dir,
        question_image,
        element_ids,
        n_shot_imgs,
        eg_per_img,
        examples_as_multiturn,
    )

    # Call the model
    if examples_as_multiturn:
        response = await call_model(prompt_or_messages, model=model_name, cache=cache)
    else:
        response = await call_model(prompt_or_messages, image_paths, model_name, cache=cache)

    # Extract answers for each element ID
    answers = []
    for element_id in element_ids:
        answer = extract_answer(response, element_id)
        answers.append(answer)

    # Load the DataFrame
    df = load_csv(csv_path)

    # Update config with model info
    config["model_name"] = model_name

    # Save the results
    df = save_results(df, prompt_or_messages, image_paths, element_ids, response, answers, config)

    return answers, df


async def run_evaluation(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_images,
    element_ids_by_image,
    model_names,
    n_shot_imgs=3,
    eg_per_img=3,
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
    # Load the DataFrame
    df = load_csv(csv_path)

    # Results by model
    results_by_model = {}
    tasks = []  # List to hold asyncio tasks

    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")

        # Create a copy of the DataFrame for this model's results
        model_df = df.copy()
        latest_result_column = None
        for img_path in question_images[:2]:
            all_element_ids = element_ids_by_image.get(img_path, [])
            if not all_element_ids:
                continue

            print(f"Processing image: {os.path.basename(img_path)}")
            print(f"Element IDs: {all_element_ids}")

            for start in range(0, len(all_element_ids), n_element_ids):
                element_ids = all_element_ids[start : start + n_element_ids]
                print(f"Batch Element IDs: {element_ids}")
                # Process all element IDs for this image
                preds, _model_df = await process_element_ids(
                    text_prompt_path,
                    csv_path,
                    eval_dir,
                    img_path,
                    element_ids,
                    model_name,
                    n_shot_imgs,
                    eg_per_img,
                    examples_as_multiturn,
                )
                answer_col = _model_df.columns[-1]
                if latest_result_column is None:
                    latest_result_column = answer_col
                    model_df[answer_col] = model_df[answer_col]
                assert (
                    latest_result_column == answer_col
                ), f"Same run_eval had different col {latest_result_column} {answer_col}"

                assembly_id, page_id = extract_assembly_and_page_from_filename(img_path)
                for answer, element_id in zip(preds, element_ids):
                    # store real
                    row = (
                        (model_df["Assembly ID"] == assembly_id)
                        & (model_df["Page ID"] == page_id)
                        & (model_df["Element ID"] == element_id)
                    )
                    model_df[row, latest_result_column] = preds
                    if True:
                        element_details = get_element_details(
                            csv_path,
                            element_id,
                            assembly_id=assembly_id,
                            page_id=page_id,
                        )
                        real_answer = element_details["Specification"]
                        print(
                            f"{evaluate_answer(answer, real_answer)} Real: `{real_answer}`,"
                            f" Model: `{answer}`"
                        )

                # If we need multiple completions, get them and use the best one
                if num_completions > 1:
                    assert False, "Doesn't combine in smart way, or use saving code"
                    for i, element_id in enumerate(element_ids):
                        all_answers = [preds[i]]

                        # Get additional completions
                        for j in range(1, num_completions):
                            answer, _ = await process_element_id(
                                text_prompt_path,
                                csv_path,
                                eval_dir,
                                img_path,
                                element_id,
                                model_name,
                                n_shot_imgs,
                                eg_per_img,
                                examples_as_multiturn,
                                cache=False,
                            )
                            all_answers.append(answer)

                        # Update the DataFrame with all answers as a list
                        # Find the row with this element ID
                        mask = model_df["Element ID"] == element_id
                        if mask.any():
                            # Get the current result column (should be the last one added)
                            result_cols = [
                                col for col in model_df.columns if col.startswith("Specification ")
                            ]
                            if result_cols:
                                model_df.loc[mask, result_cols[-1]] = str(
                                    all_answers
                                )  # Convert to string for DataFrame storage

            try:
                # Calculate metrics
                metrics = calculate_metrics(model_df, latest_result_column)
                results_by_model[model_name] = metrics

                # Save the DataFrame
                output_path = f"data/results/{latest_result_column.replace(' ', '_')}.csv"
                save_df_with_results(model_df, latest_result_column, output_path)

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
        "--image",
        type=str,
        default=None,  # "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg1.png",
        help="Path to the question image",
    )
    parser.add_argument("--element_id", type=str, default=None, help="Element ID to query")
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
    args = parser.parse_args()

    # Process a single element ID
    if not args.eval_all and args.element_id and args.image:
        answer, df = await process_element_id(
            args.prompt,
            args.csv,
            args.eval_dir,
            args.image,
            args.element_id,
            args.model,
            args.n_shot_imgs,
            args.eg_per_img,
            args.multiturn,
        )
        print(f"Answer for {args.element_id}: {answer}")

    # Run evaluation on all element IDs in the CSV
    elif args.eval_all:
        df = load_csv(args.csv)

        # Group element IDs by image
        element_ids_by_image = {}
        for _, row in df.iterrows():
            if pd.isna(row["Assembly ID"]) or pd.isna(row["Page ID"]) or pd.isna(row["Element ID"]):
                continue

            assembly_id = int(row["Assembly ID"])
            page_id = int(row["Page ID"])
            element_id = row["Element ID"]

            # Find the corresponding image
            image_path = None
            for file_name in os.listdir(args.eval_dir):
                if (
                    f"nist_ftc_{assembly_id}" in file_name
                    and f"_pg{page_id}" in file_name
                    and file_name.endswith(".png")
                ):
                    image_path = os.path.join(args.eval_dir, file_name)
                    break

            if image_path:
                if image_path not in element_ids_by_image:
                    element_ids_by_image[image_path] = []
                element_ids_by_image[image_path].append(element_id)

        # Run evaluation
        model_names = [args.model]  # You can add more models here if needed
        results = await run_evaluation(
            args.prompt,
            args.csv,
            args.eval_dir,
            list(element_ids_by_image.keys()),
            element_ids_by_image,
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

    else:
        print(
            "Please specify an element_id(s) and image or use --eval_all to evaluate all elements"
        )


if __name__ == "__main__":
    asyncio.run(main())
