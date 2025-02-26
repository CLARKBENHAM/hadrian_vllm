# prompt_generator.py
import os
import random
import re
import json
import hashlib
from datetime import datetime

from hadrian_vllm.utils import extract_assembly_and_page_from_filename, get_git_hash, get_current_datetime


def load_prompt_template(prompt_path):
    """Load the prompt template from the given path."""
    with open(prompt_path, "r") as f:
        return f.read()


def extract_element_ids_from_image(csv_path, image_path):
    """
    Extract element IDs from an image using the CSV data.

    Args:
        csv_path: Path to the CSV with GD&T data
        image_path: Path to the image

    Returns:
        List of element IDs for this image
    """
    # Extract assembly ID and page ID from the image filename
    basename = os.path.basename(image_path)
    assembly_id, page_id = extract_assembly_and_page_from_filename(basename)

    if assembly_id is None or page_id is None:
        return []

    # Load the CSV
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Filter by assembly ID and page ID
    filtered_df = df[(df["Assembly ID"] == assembly_id) & (df["Page ID"] == page_id)]

    # Get unique element IDs
    element_ids = filtered_df["Element ID"].unique().tolist()

    # Filter out None or NaN values
    element_ids = [eid for eid in element_ids if pd.notna(eid) and eid]

    return element_ids


def select_few_shot_examples(csv_path, eval_dir, question_image, n_shot=3, eg_per_img=3):
    """
    Select n_shot/eg_per_img random images from eval_dir (excluding the question_image)
    and eg_per_img random element IDs for each image.

    Returns:
        List of tuples (image_path, element_ids_list)
    """
    # Set random seed to 0 for reproducibility
    random.seed(0)

    # Get all image files from the directory
    image_files = [f for f in os.listdir(eval_dir) if f.endswith(".png")]

    # Exclude the question image
    question_image_basename = os.path.basename(question_image) if question_image else None
    available_images = [img for img in image_files if img != question_image_basename]

    # If we don't have enough images, use what we have
    n_images = min(int(n_shot / eg_per_img) if eg_per_img > 0 else n_shot, len(available_images))
    if n_images == 0 and available_images:
        n_images = 1

    if not available_images or n_images == 0:
        return []

    # Select random images
    selected_images = random.sample(available_images, n_images)
    results = []

    for img in selected_images:
        img_path = os.path.join(eval_dir, img)
        # Get element IDs for this image
        element_ids = extract_element_ids_from_image(csv_path, img_path)

        # If we have element IDs, select random ones
        if element_ids:
            num_examples = min(eg_per_img, len(element_ids))
            selected_ids = random.sample(element_ids, num_examples)
            results.append((img_path, selected_ids))

    return results


def get_example_answers(csv_path, img_path, element_ids=None):
    """
    Get GD&T data for the given element IDs from the CSV.

    Returns:
        Dictionary mapping element IDs to their GD&T data
    """
    import pandas as pd

    # Extract assembly ID and page ID from the image filename
    basename = os.path.basename(img_path)
    assembly_id, page_id = extract_assembly_and_page_from_filename(basename)

    if assembly_id is None or page_id is None:
        return {}

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter by assembly ID and page ID
    filtered_df = df[(df["Assembly ID"] == assembly_id) & (df["Page ID"] == page_id)]
    if element_ids is None:
        element_ids = filtered_df["Element ID"].to_list()

    # Create a dictionary of element IDs to GD&T data
    element_to_spec = {}
    for _, row in filtered_df.iterrows():
        element_id = row["Element ID"]
        if element_id in element_ids:
            assert pd.notna(row["Specification"])
            element_to_spec[element_id] = row["Specification"]

    return element_to_spec


def generate_few_shot_prompt(prompt_template, examples, question_image, question_ids):
    """
    Generate a few-shot prompt with examples and the target question.

    Args:
        prompt_template: The prompt template string
        examples: List of tuples (image_path, element_ids, element_to_spec)
        question_image: Path to the image containing the target element ID
        question_ids: List of element IDs to extract GD&T data for

    Returns:
        Generated prompt
    """
    # Generate few-shot examples text
    few_shot_text = "Examples\n"
    for i, (_, element_ids, element_to_spec) in enumerate(examples):
        img_num = i + 1

        for element_id in element_ids:
            if element_id in element_to_spec:
                spec = element_to_spec[element_id]
                few_shot_text += f"Img{img_num}:\n{element_id}: <answer>{spec}<answer>\n\n"

    # Add the question
    question_text = "Question:\n"
    img_num = len(examples) + 1

    for element_id in question_ids:
        question_text += f"Img{img_num}:\n{element_id}:\n"

    # Replace placeholders in the template
    prompt = re.sub(r"\{\{\{Example\}\}\}", few_shot_text.strip(), prompt_template, flags=re.DOTALL)
    prompt = re.sub(r"\{\{\{Question\}\}\}", question_text.strip(), prompt, flags=re.DOTALL)

    return prompt


def generate_multiturn_messages(prompt_template, examples, question_image, question_ids):
    """
    Generate messages for a multi-turn conversation with examples and the target question.

    Args:
        prompt_template: The system prompt is assumed to be prompt_template up till {{{Example}}}
        examples: List of tuples (image_path, element_ids, element_to_spec)
        question_image: Path to the image containing the target element ID
        question_ids: List of element IDs to extract GD&T data for

    Returns:
        List of message objects
    """
    ix = prompt_template.find("{{{Example}}}")
    if ix > 0:
        system_prompt = prompt_template[:ix]
    else:
        system_prompt = prompt_template
    messages = [{"role": "system", "content": system_prompt}]

    # Add few-shot examples as separate turns
    for i, (img_path, element_ids, element_to_spec) in enumerate(examples):
        img_num = i + 1

        for element_id in element_ids:
            if element_id in element_to_spec:
                # Add user message with the example question
                user_message = f"Img{img_num}:\n{element_id}:"
                messages.append({"role": "user", "content": user_message, "image_path": img_path})

                # Add assistant message with the example answer
                spec = element_to_spec[element_id]
                assistant_message = f"<answer>{spec}<answer>"
                messages.append({"role": "assistant", "content": assistant_message})

    # Add the actual question
    img_num = len(examples) + 1
    for element_id in question_ids:
        user_message = f"Img{img_num}:\n{element_id}:"
        messages.append({"role": "user", "content": user_message, "image_path": question_image})

    return messages


def element_ids_per_img_few_shot(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_image,
    question_ids,
    n_shot=3,
    eg_per_img=3,
    examples_as_multiturn=False,
):
    """
    Generate a few-shot prompt with examples and the target question.

    Args:
        text_prompt_path: Path to the text prompt template
        csv_path: Path to the CSV with GD&T data
        eval_dir: Directory containing the evaluation images
        question_image: Path to the image containing the target element ID
        question_ids: List of element IDs to extract GD&T data for
        n_shot: Number of few-shot examples
        eg_per_img: Number of examples per image
        examples_as_multiturn: Whether to format examples as multiple turns

    Returns:
        Tuple of (generated_prompt or messages, ordered_image_paths, config)
    """
    # Load the prompt template
    prompt_template = load_prompt_template(text_prompt_path)

    # Select few-shot examples
    example_tuples = select_few_shot_examples(
        csv_path, eval_dir, question_image, n_shot, eg_per_img
    )

    # Get GD&T data for these examples
    examples = []
    image_paths = []

    for img_path, element_ids in example_tuples:
        image_paths.append(img_path)
        element_to_spec = get_example_answers(csv_path, img_path, element_ids)
        examples.append((img_path, element_ids, element_to_spec))

    # Add the question image
    if question_image and question_image not in image_paths:
        image_paths.append(question_image)

    # Create the configuration object
    config = {
        "prompt_path": text_prompt_path,
        "n_few_shot": n_shot,
        "eg_per_img": eg_per_img,
        "few_shot_examples": examples,
        "examples_as_multiturn": examples_as_multiturn,
        "datetime": get_current_datetime(),
        "git_hash": get_git_hash(),
        "question_ids": question_ids if isinstance(question_ids, list) else [question_ids],
        "question_image": os.path.basename(question_image) if question_image else None,
    }

    if examples_as_multiturn:
        # Create messages for multi-turn conversation
        messages = generate_multiturn_messages(
            prompt_template,
            examples,
            question_image,
            question_ids if isinstance(question_ids, list) else [question_ids],
        )
        config["messages"] = messages
        # [ # want to send along image paths
        #     {k: v for k, v in msg.items() if k != "image_path"} for msg in messages
        # ]
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config["hash"] = config_hash
        config["result_column"] = f"Specification {config_hash}"
        return messages, image_paths, config
    else:
        # Generate a standard prompt
        prompt = generate_few_shot_prompt(
            prompt_template,
            examples,
            question_image,
            question_ids if isinstance(question_ids, list) else [question_ids],
        )
        config["generated_prompt"] = prompt
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config["hash"] = config_hash
        config["result_column"] = f"Specification {config_hash}"
        return prompt, image_paths, config
