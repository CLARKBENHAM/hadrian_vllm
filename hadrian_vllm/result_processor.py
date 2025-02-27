# result_processor.py
import re
import json
import os
from datetime import datetime
import pandas as pd

from hadrian_vllm.utils import extract_assembly_and_page_from_filename


def extract_answer(response, element_id=None):
    """
    Extract the answer from the model's response.

    Args:
        response: The full response from the model
        element_id: The element ID we were asking about (optional)

    Returns:
            The extracted answer or None if not found
    """
    # Try to find content between <answer> tags
    pattern = r"<answer>(.*?)</*answer>"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        if element_id and len(matches) > 1:
            # If multiple answers and we have an element ID, try to find the right one
            for i, match in enumerate(matches):
                # Look at nearby context to find the element ID
                start_pos = response.find(f"<answer>{match}<answer>")
                context_start = max(0, start_pos - 100)
                context = response[context_start:start_pos]
                if element_id in context:
                    return match.strip()
            # If no specific match found, return the first one
            return matches[0].strip()
        else:
            # Return the first match
            return matches[0].strip()

    return None


def save_results(
    df,
    prompt_or_messages,
    image_paths,
    question_ids,
    full_response,
    extracted_answers,
    config,
    config_dir="data/prompt_config",
    completion_dir="data/llm_completions",
):
    """
    Save the results to files and update the DataFrame.

    Args:
        df: The DataFrame with GD&T data
        prompt_or_messages: The generated prompt or messages
        image_paths: List of image paths used
        question_ids: List of element IDs queried
        full_response: The full response from the model
        extracted_answers: List of extracted answers
        config: Configuration object with metadata
        config_dir: Directory to save config files
        completion_dir: Directory to save completion files

    Returns:
            Updated DataFrame with results
    """
    # Create directories if they don't exist
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(completion_dir, exist_ok=True)

    # Get the hash from the config
    config_hash = config["hash"]
    result_col = config["result_column"]

    # Save config
    with open(os.path.join(config_dir, f"{config_hash}.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save completion
    completion_data = {
        "config_hash": config_hash,
        "element_ids": question_ids if isinstance(question_ids, list) else [question_ids],
        "question_images": image_paths,
        "full_response": full_response,
        "extracted_answer": (
            extracted_answers if isinstance(extracted_answers, list) else [extracted_answers]
        ),
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(completion_dir, f"{config_hash}.json"), "w") as f:
        json.dump(completion_data, f, indent=2)

    # Add results to DataFrame
    q_img = image_paths[-1]
    assembly_id, page_id = extract_assembly_and_page_from_filename(q_img)
    if isinstance(question_ids, list):
        # make answer a dict
        assert len(extracted_answers) == len(question_ids), f"{extracted_answers}, {question_ids}"
        for answer, element_id in zip(extracted_answers, question_ids):
            # Find the row with this element ID
            mask = (
                (df["Element ID"] == element_id)
                & (df["Page ID"] == page_id)
                & (df["Assembly ID"] == assembly_id)
            )
            assert (
                mask.sum() == 1
            ), f"Duplicate rows {element_id} {page_id} {assembly_id} {config_hash}"
            df.loc[mask, result_col] = answer
    else:
        # Single element ID case
        mask = (
            (df["Element ID"] == question_ids)
            & (df["Page ID"] == page_id)
            & (df["Assembly ID"] == assembly_id)
        )
        assert (
            mask.sum() == 1
        ), f"Duplicate rows {question_ids} {page_id} {assembly_id} {config_hash}"
        df.loc[mask, result_col] = extracted_answers

    return df
