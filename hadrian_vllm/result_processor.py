# %%
# # result_processor.py
import re
import json
import os
from datetime import datetime

import logging

from hadrian_vllm.utils import extract_assembly_and_page_from_filename, is_debug_mode

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_code_block(response):
    """Extract content from code blocks in the response"""
    # Try to find content inside a code block
    code_block_pattern = r"```(?:json|text)?\s*([\s\S]*?)```"
    code_matches = re.findall(code_block_pattern, response)

    if code_matches:
        return code_matches[0].strip()
    return None


def extract_json_dict(text):
    """Try to parse text as JSON and return a dictionary"""
    try:
        # Try parsing directly
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        # Try finding JSON object in the text
        json_pattern = r"\{[\s\S]*\}"
        json_match = re.search(json_pattern, text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return None


def extract_element_id_lines(text):
    """Extract lines that match the pattern 'ElementID: Value'"""
    element_pattern = r"([A-Za-z0-9\-]+):\s*(.*?)(?=\n[A-Za-z0-9\-]+:|$)"
    matches = re.findall(element_pattern, text)
    return {elem.strip(): value.strip() for elem, value in matches}


def extract_answers_from_text(response, element_ids):
    """
    Extract answers for specific element IDs from text response.

    Args:
        response: The text response from the model
        element_ids: List of element IDs to extract answers for

    Returns:
        List of answers in the same order as element_ids
    """
    # Try to find content inside a code block first
    code_content = parse_code_block(response)
    if code_content:
        # Try parsing as JSON
        json_dict = extract_json_dict(code_content)
        if json_dict:
            # {element_id: value} If we have a JSON dictionary, try to get answers for each element ID
            return [json_dict.get(elem_id, None) for elem_id in element_ids]

        # If not JSON, try to extract element ID lines from the code block
        element_dict = extract_element_id_lines(code_content)
        if element_dict:
            return [element_dict.get(elem_id, None) for elem_id in element_ids]

    # Try to extract element ID lines from the entire response
    element_dict = extract_element_id_lines(response)
    if element_dict:
        return [element_dict.get(elem_id, None) for elem_id in element_ids]

    # If all else fails, try to find answers in <answer> tags
    answers = []
    for elem_id in element_ids:
        # Try to find a section with this element ID followed by an <answer> tag
        pattern = rf"{re.escape(elem_id)}[^\n]*?<answer>(.*?)<answer>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            answers.append(match.group(1).strip())
        else:
            answers.append(None)

    return answers


# TODO this is screwy and could be cleaned up
def extract_answer(response, element_ids=None):
    """
    Extract the answer(s) from the model's response.

    Args:
        response: The full response from the model
        element_ids: Single element ID or list of element IDs to extract answers for

    Returns:
        The extracted answer(s): single answer if element_ids is a string,
        or list of answers in the same order as element_ids if it's a list
    """
    # Handle the case where element_ids is a single string
    if isinstance(element_ids, str):
        # For a single element ID, use the <answer> tag pattern first
        pattern = r"<answer>(.*?)</*answer>"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Look for a match specifically for this element ID
            for match in matches:
                # Look at nearby context to find the element ID
                start_pos = response.find(f"<answer>{match}<answer>")
                context_start = max(0, start_pos - 100)
                context = response[context_start:start_pos]
                if element_ids in context:
                    return match.strip()

            # If no specific match found, return the first one
            return matches[0].strip()

        # If no <answer> tags found, try other methods
        answers = extract_answers_from_text(response, [element_ids])
        if answers and answers[0]:
            return answers[0]

        # If all else fails, return None
        return None

    # Handle the case where element_ids is a list
    elif isinstance(element_ids, list):
        # Try to extract answers for all element IDs
        answers = extract_answers_from_text(response, element_ids)

        # Count how many answers we found
        found_answers = sum(1 for a in answers if a is not None)

        # If we didn't find all answers, log a warning
        if found_answers < len(element_ids):
            logger.warning(
                f"Found only {found_answers} out of {len(element_ids)} answers in the response."
            )

        return answers

    # Handle the case where element_ids is None
    else:
        # Just try to find any <answer> tags
        pattern = r"<answer>(.*?)</*answer>"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no <answer> tags, try to extract from code blocks or formatted text
        code_content = parse_code_block(response)
        if code_content:
            return code_content

        # If all else fails, return the entire response
        return response


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

    # Ensure question_ids is a list and extracted_answers matches
    if not isinstance(question_ids, list):
        question_ids = [question_ids]
        if not isinstance(extracted_answers, list):
            extracted_answers = [extracted_answers]
    elif not isinstance(extracted_answers, list):
        logger.warn(
            "Question IDs as list but single answer str, duplicate it"
            f" {question_ids} {extract_answer} {config_hash}"
        )
        extracted_answers = [extracted_answers] * len(question_ids)

    # Make sure extracted_answers and question_ids have the same length
    if len(extracted_answers) < len(question_ids):
        logger.warning(
            f"Fewer answers ({len(extracted_answers)}) than questions ({len(question_ids)})."
            f" Padding with None. {config_hash}"
        )
        extracted_answers = extracted_answers + [None] * (
            len(question_ids) - len(extracted_answers)
        )
    elif len(extracted_answers) > len(question_ids):
        logger.warning(
            f"More answers ({len(extracted_answers)}) than questions ({len(question_ids)})."
            f" Truncating. {config_hash}"
        )
        extracted_answers = extracted_answers[: len(question_ids)]

    # Save completion
    completion_data = {
        "config_hash": config_hash,
        "element_ids": question_ids,
        "question_images": image_paths,
        "full_response": full_response,
        "extracted_answer": extracted_answers,
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
            if is_debug_mode():
                assert answer is not None, element_id
            mask = (
                (df["Element ID"] == element_id)
                & (df["Page ID"] == page_id)
                & (df["Assembly ID"] == assembly_id)
            )
            if mask.sum() != 1 and not (element_id in ("T17", "T18") and assembly_id == 6):
                print(
                    f"WARN n rows {mask.sum()}!=1 for"
                    f" {element_id} {page_id} {assembly_id} {config_hash}"
                )
            df.loc[mask, result_col] = answer
    return df


if __name__ == "__main__":
    print(
        extract_answer(
            '```json\n{\n"element_id": "DF1",\n"gdt_data": "Datum Feature Symbol A"\n}\n```', "DF1"
        )
    )
