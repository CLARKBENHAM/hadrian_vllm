# prompt_generator.py
import os
import random
import re
import json
import hashlib
from datetime import datetime
import logging

from hadrian_vllm.utils import (
    extract_assembly_and_page_from_filename,
    get_git_hash,
    get_current_datetime,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    assembly_id, page_id = extract_assembly_and_page_from_filename(image_path)

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


def select_few_shot_examples(
    csv_path, eval_dir, question_image, question_ids_count=1, n_shot_imgs=3, eg_per_img=3
):
    """
    Select n_shot_imgs random images from eval_dir (excluding the question_image)
    and eg_per_img lists of lentgth queston_ids_count element IDs for each image.

    Returns:
        List of tuples (image_path, element_ids_list)
    """
    # Set random seed to 0 for reproducibility
    random.seed(0)

    # Get all image files from the directory
    image_files = [f for f in os.listdir(eval_dir) if f.endswith(".png")]

    # Exclude the question image
    question_image_basename = os.path.basename(question_image) if question_image else None
    _available_images = [img for img in image_files if img != question_image_basename]
    available_images = [
        img
        for img in image_files
        if (img.split(".")[0] not in question_image)
        and (
            extract_assembly_and_page_from_filename(img)
            != extract_assembly_and_page_from_filename(question_image)
        )
    ]
    if _available_images != available_images:
        logger.debug(f"dropped extra images {set(_available_images) - set(available_images)}")

    # If we don't have enough images, use what we have
    n_images = min(n_shot_imgs, len(available_images))
    if n_images == 0:
        return []

    # Select random images
    selected_images = random.sample(available_images, n_images)
    results = []
    for img in selected_images:
        img_path = os.path.join(eval_dir, img)
        # Get all element IDs for this image
        all_element_ids = extract_element_ids_from_image(csv_path, img_path)

        # If we have enough element IDs, create eg_per_img examples with question_ids_count IDs each
        if len(all_element_ids) >= question_ids_count:
            for _ in range(0, min(eg_per_img, len(all_element_ids)), question_ids_count):
                selected_ids = random.sample(all_element_ids, question_ids_count)
                all_element_ids = [e for e in all_element_ids if e not in selected_ids]
                results.append((img_path, selected_ids))
        else:
            # If not enough IDs will have duplicates
            for _ in range(eg_per_img):
                selected_ids = random.choices(all_element_ids, k=question_ids_count)
                if selected_ids:
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
    assembly_id, page_id = extract_assembly_and_page_from_filename(img_path)

    if assembly_id is None or page_id is None:
        return {}

    if not hasattr(get_example_answers, csv_path):
        setattr(get_example_answers, csv_path, pd.read_csv(csv_path))
    df = getattr(get_example_answers, csv_path)

    # Filter by assembly ID and page ID
    filtered_df = df[(df["Assembly ID"] == assembly_id) & (df["Page ID"] == page_id)]
    if element_ids is None:
        element_ids = filtered_df["Element ID"].to_list()

    assert page_id != 6, f"TEMP for hard debug only {img_path}"

    # Create a dictionary of element IDs to GD&T data
    element_to_spec = {}
    for _, row in filtered_df.iterrows():
        element_id = row["Element ID"]
        if element_id in element_ids:
            assert pd.notna(row["Specification"])
            element_to_spec[element_id] = row["Specification"]

    return element_to_spec


def ans_text(page_answers):
    ans = ", ".join(map(lambda i: f'"{i}"', page_answers))
    return (
        "The Answer is one of the elements Below. Make sure you return an entry from this"
        f" between <answer> tags. list:\n{ans}"
    )


# handles multiple correctly?
def generate_few_shot_prompt(
    prompt_template,
    examples,
    question_image,
    question_ids,
    img2page_answers={},
):
    """
    Generate a few-shot single string prompt with examples and the target question.

    Args:
        prompt_template: The prompt template string
        examples: List of tuples (image_path, element_ids_list, element_to_spec)
        question_image: Path to the image containing the target element ID
        question_ids: List of element IDs to extract GD&T data for

    Returns:
        Generated prompt
    """
    # Generate few-shot examples text
    few_shot_text = "Examples\n"
    img2i = {}
    for _, (path, element_ids, element_to_spec) in enumerate(examples):
        if path not in img2i:
            img2i[path] = len(img2i) + 1
        img_num = img2i[path]

        for element_id in element_ids:
            if element_id in element_to_spec:
                spec = element_to_spec[element_id]
                few_shot_text += f"Img{img_num}:\n{element_id}: <answer>{spec}<answer>\n"
                if "{{{all_img_possible_answer}}}" in prompt_template:
                    few_shot_text += ans_text(img2page_answers[path])

        few_shot_text += "\n"  # Add all element IDs for this image as a group

    # Add the question
    question_text = "Question:\n"
    img_num = len(examples) + 1

    for element_id in question_ids:
        question_text += f"Img{img_num}:\n{element_id}:\n"

    # Replace placeholders in the template
    prompt = re.sub(r"\{\{\{Example\}\}\}", few_shot_text.strip(), prompt_template, flags=re.DOTALL)
    prompt = re.sub(r"\{\{\{Question\}\}\}", question_text.strip(), prompt, flags=re.DOTALL)
    for t in ["{{{Answer}}}", "{{{all_img_possible_answer}}}"]:
        # use '"{{{all_img_possible_answer}}}"' instead of Answer
        if t in prompt:
            prompt = re.sub(
                t,
                ans_text(img2page_answers[question_image]),
                prompt,
                flags=re.DOTALL,
            )
    for _, es, _ in examples:
        for e in es:
            assert e in prompt, f"`{e}` not in prompt {prompt}"
    for e in question_ids:
        assert e in prompt, f"question `{e}` not in prompt {prompt}"

    assert "{{{" not in prompt, prompt
    return prompt


def generate_multiturn_messages(
    prompt_template,
    examples,
    question_image,
    question_ids,
    img2page_answers={},
):
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
    # remove single prompt replacement examples to get only system prompt
    prompt_template = re.sub("\{\{\{Example\}\}\}\S*", "", prompt_template)
    system_prompt = re.sub("\S*\{\{\{Question\}\}\}\S*", "", prompt_template)
    # deprecated, only to append answers
    add_answers = "{{{Answers}}" in system_prompt
    system_prompt = re.sub("\S*\{\{\{Answers\}\}\}\S*", "", system_prompt)
    # Answers provided for all few shot cases
    add_all_answers = "{{{all_img_possible_answer}}}" in system_prompt
    system_prompt = system_prompt.replace("{{{all_img_possible_answer}}}", "")

    assert "{{{" not in system_prompt, system_prompt

    messages = [{"role": "system", "content": system_prompt}]
    sent_images = set()

    # Add few-shot examples as separate turns
    img_num = 0
    for i, (img_path, element_ids, element_to_spec) in enumerate(examples):

        for element_id in element_ids:
            if element_id in element_to_spec:
                # Add user message with all element IDs
                # Only include image if it hasn't been sent yet.
                message_data = {"role": "user"}
                if img_path not in sent_images:
                    img_num += 1
                    user_message = f"Img{img_num}:\n"
                    message_data["image_path"] = img_path
                    sent_images.add(img_path)
                else:
                    # Optionally, prepend a note to indicate that it refers to the previous image.
                    user_message = f"the previous image\nImg{img_num}:\n"
                for element_id in element_ids:
                    user_message += f"{element_id}:\n"
                # all possible ans for this q
                if add_all_answers:
                    user_message += ans_text(img2page_answers[img_path])
                message_data["content"] = user_message
                messages.append(message_data)

                # Add assistant message with all answers
                assistant_message = ""
                for element_id in element_ids:
                    if element_id in element_to_spec:
                        spec = element_to_spec[element_id]
                        assistant_message += f"{element_id}: <answer>{spec}<answer>\n"

                messages.append({"role": "assistant", "content": assistant_message})

    # Add the actual question
    img_num = len(sent_images) + 1
    user_message = f"\nImg{img_num}:\n"
    for element_id in question_ids:
        user_message += f"{element_id}:\n"
    if add_answers or add_all_answers:
        user_message += ans_text(img2page_answers[question_image])
    # # trials for hard
    # user_message += (
    #     "The examples above were from the full image. Since question was a little harder I have"
    #     " blanked out all the other element IDs and highlighted the text on the image within a"
    #     " bright red Oval. Look at the element ID and oval and return the appropriate text. There's"
    #     " been a lot of problems with returning the text from multiple different elemnent IDs. Just"
    #     " return the single line of text, or half line if there's 2 element IDs on either side of"
    #     " the line. Do not return more symbols than those. Still return the answer within <answer>"
    #     " tags. You got it this time! I really believe you're going to crush it."
    # )
    # # This didn't go better
    # user_message += (
    #     "The examples above were from the full image. Since question was a little harder I have"
    #     " blanked out all the other element IDs. So there's only 1 element ID to return from this"
    #     " question. There's been a lot of problems with returning the text from multiple different"
    #     " elemnent IDs. Just return the single line of text, or half line if there's 2 element IDs"
    #     " on either side of the line. Do not return more symbols than those. Also make sure '⌴'"
    #     " doesnt get transcribed as 'L⌴'.  Still return the answer within <answer> tags. You got it"
    #     " this time! I really believe you're going to crush it."
    # )
    messages.append({"role": "user", "content": user_message, "image_path": question_image})

    return messages


def element_ids_per_img_few_shot(
    text_prompt_path,
    csv_path,
    eval_dir,
    question_image,
    question_ids,
    n_shot_imgs=3,
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
        n_shot_imgs: Number of few-shot images
        eg_per_img: Number of examples per image
            total example ids: n_shot_imgs*eg_per_img*len(element_ids)
        examples_as_multiturn: Whether to format examples as multiple turns

    Returns:
        Tuple of (generated_prompt or messages, ordered_image_paths, config)
    """
    # Load the prompt template
    prompt_template = load_prompt_template(text_prompt_path)

    # Select few-shot examples
    example_tuples = select_few_shot_examples(
        csv_path, eval_dir, question_image, len(question_ids), n_shot_imgs, eg_per_img
    )

    # Get GD&T data for these examples
    examples = []
    image_paths = []

    for img_path, element_ids in example_tuples:
        image_paths.append(img_path)
        element_to_spec = get_example_answers(csv_path, img_path, element_ids)
        examples.append((img_path, element_ids, element_to_spec))

    # Add the question image
    assert question_image not in image_paths
    image_paths.append(question_image)

    # Create the configuration object
    shared_config = {  # what all element_ids for this evaluation suite will use
        "prompt_path": text_prompt_path,
        "n_few_shot": n_shot_imgs,
        "eg_per_img": eg_per_img,
        "git_hash": get_git_hash(),
        "examples_as_multiturn": examples_as_multiturn,
    }

    shared_config_str = json.dumps(shared_config, sort_keys=True)
    shared_hash = hashlib.md5(shared_config_str.encode()).hexdigest()[:8]
    config = {
        **shared_config,
        # specific to this call
        "datetime": get_current_datetime(),
        "few_shot_examples": examples,
        "question_ids": question_ids if isinstance(question_ids, list) else [question_ids],
        "question_image": question_image,
        "result_column": f"Specification {shared_hash}",
    }

    img2page_answers = {}
    for img_path, _, _ in examples:
        img2page_answers[img_path] = list(sorted(get_example_answers(csv_path, img_path).values()))
    img2page_answers[question_image] = list(
        sorted(get_example_answers(csv_path, question_image).values())
    )

    if examples_as_multiturn:
        # Create messages for multi-turn conversation
        messages = generate_multiturn_messages(
            prompt_template,
            examples,
            question_image,
            question_ids if isinstance(question_ids, list) else [question_ids],
            img2page_answers=img2page_answers,
        )
        config["messages"] = messages
        # [ # want to send along image paths
        #     {k: v for k, v in msg.items() if k != "image_path"} for msg in messages
        # ]
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config["hash"] = config_hash
        return messages, image_paths, config
    else:
        # Generate a standard prompt
        prompt = generate_few_shot_prompt(
            prompt_template,
            examples,
            question_image,
            question_ids if isinstance(question_ids, list) else [question_ids],
            img2page_answers=img2page_answers,
        )
        config["generated_prompt"] = prompt
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config["hash"] = config_hash
        # only send 1 image if multiple examples use same image, preseving order
        # while in multiturn each turn gets its own image as seperate message
        u_paths = []
        for i in image_paths:
            if i not in u_paths:
                u_paths += [i]
        return prompt, u_paths, config
