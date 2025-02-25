# %%
import time
import os
import re
import io
import json
import asyncio
import base64
from collections import deque
import math

import pandas as pd
from PIL import Image
import openai

import nest_asyncio

nest_asyncio.apply()

from google import genai
from google.genai import types

from src.image_cost import validate_cost

# Initialize genai with your API key - IMPORTANT!
# genai.configure(
#     api_key=os.environ["GEMINI_API_KEY"]
# )  # Ensure GEMINI_API_KEY is set in your environment

RATE_LIMITS = {  # per minute
    "o1": 500,
    "o3-mini": 5000,
    "gpt-4o": 10000,
    "gemini-2.0-pro-exp-02-05": 5,
    "gemini-2.0-flash-001": 2000,
}
# Request tracking for each model
model_request_timestamps = {
    model_name: deque(maxlen=RATE_LIMITS[model_name]) for model_name in RATE_LIMITS
}

# Lock for accessing the timestamps safely in async context
model_locks = {model_name: asyncio.Lock() for model_name in RATE_LIMITS}

# Initial Startup Delay (Seconds), delay between sequential model calls
# inter_request_delay =  0.3 * 60 / max(RATE_LIMITS.values())
# I'll assume there's enough contention getting the single model_lock too not start too many all at once


def get_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_base64_bytes(image_path: str) -> bytes:  # Return type as bytes
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read())  # Return encoded bytes directly


async def lock_model(model):
    # Check if we need to wait before making this request
    async with model_locks[model]:
        current_time = time.time()
        # If we've reached the limit of requests per minute
        if len(model_request_timestamps[model]) >= RATE_LIMITS[model]:
            # Calculate how long until a slot opens up (oldest request + 60 seconds)
            oldest_request = model_request_timestamps[model][0]
            time_until_available = oldest_request + 60
            await asyncio.sleep(time_until_available)
        # Add this request's timestamp
        model_request_timestamps[model].append(current_time)


async def generate_gemini(prompt, base64_bytes, model="gemini-2.0-flash-001"):
    client = genai.Client(
        vertexai=True,
        project="gen-lang-client-0392240747",
        location="us-central1",
    )

    image = types.Part.from_bytes(
        data=base64.b64decode(base64_bytes),
        mime_type="image/png",
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                image,
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=1,
        seed=0,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
    )

    await lock_model(model)
    o = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            o += chunk.text
        return o
    except Exception as api_error:
        print(f"Gemini API Error: {api_error}")
        return f"Gemini API Error: {api_error}"


async def find_element_ids_on_page(image_path: str, model: str) -> list:
    if model in ("gemini-2.0-flash-001", "gemini-2.0-pro-exp-02-05"):
        base64_bytes = get_base64_bytes(image_path)
        prompt = (
            "You are an expert in reading mechanical drawings and identifying Element IDs.\nGiven"
            " an image, find all Element IDs and return them as a JSON array.\nElement IDs are"
            " typically short alphanumeric strings like 'T1', 'D1', 'DF1', etc.\nCarefully examine"
            " the entire image to identify every single Element ID.\n\n**IMPORTANT:** Your ENTIRE"
            " response MUST be valid JSON, enclosed within `<answer></answer>` tags.\nDo not"
            " include any introductory or explanatory text outside the `<answer>` tags.\nFor"
            ' example, a correct response would look like:\n`<answer>["ID1", "ID2",'
            ' "ID3"]</answer>`\n\nIf no Element IDs are found, return an empty JSON array like'
            " this:\n`<answer>[]</answer>`\n\nBegin!"
        )
        reply_content = await generate_gemini(prompt, base64_bytes)
    elif model in ("o1", "gpt-4o"):
        prompt = (
            'You are given an image with mechanical drawings.\nReturn a JSON array of all "Element'
            ' IDs" visible, e.g. ["T1", "D1", "DF1"]. Go over the image slowly and carefully to'
            " return every last one.\nBe sure to only respond with valid JSON within the"
            " <answer></answer> tags but you're allowed to think outside of those to ensure the"
            " highest accuracy."
        )
        image_base64 = get_base64(image_path)
        await lock_model(model)
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model=model,
            max_completion_tokens=10000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        reply_content = response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported model {model}")
    print("Response from", model, "for", image_path)
    print(reply_content)
    match = re.search(r"<answer>(.*?)</answer>", reply_content, re.DOTALL)
    if match:
        json_data = match.group(1).strip()
        try:
            element_ids = json.loads(json_data)
            if isinstance(element_ids, list):
                return element_ids
            else:
                print("Extracted JSON is not a list.")
        except json.JSONDecodeError:
            print("Failed to decode JSON:", json_data)
    else:
        print("No <answer> tags found in the response.")
    return []


def compute_metrics(pred_mapping, csv_labeling):
    hallucinations = 0
    missed = 0
    disagreements = 0
    common_count = 0
    for key, pages in pred_mapping.items():
        if key not in csv_labeling:
            hallucinations += 1
        else:
            common_count += 1
            if csv_labeling[key] not in pages:
                disagreements += 1
    for key in csv_labeling:
        if key not in pred_mapping:
            missed += 1
    return hallucinations, missed, disagreements, common_count


if __name__ == "__main__":
    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    image_dir = "data/eval_on/single_images/"
    df = pd.read_csv(csv_path)
    csv_labeling = {}
    for _, row in df.iterrows():
        assembly = row["Assembly ID"]
        elem = row["Element ID"]
        page = str(row["Page ID"]).strip()
        csv_labeling[(assembly, elem)] = page

    tasks = {}  # key: (model, assembly, page) -> list of tasks
    image_files = []
    for file_name in sorted(os.listdir(image_dir)):
        if not file_name.endswith(".png"):
            continue
        m = re.search(r"nist_ftc_(\d+)", file_name)
        if not m:
            continue
        assembly = int(m.group(1))
        page_str = file_name.split("_pg")[-1].replace(".png", "").strip()
        filepath = os.path.join(image_dir, file_name)
        image_files.append((filepath, assembly, page_str))
    os.makedirs("data/eval_on/quarter_images/", exist_ok=True)
    models = ["gemini-2.0-flash-001", "gpt-4o", "o1"]
    validate_cost(image_files, models)
    for filepath, assembly, page_str in image_files:
        try:
            image = Image.open(filepath)
        except Exception as e:
            print(f"Error opening image {filepath}: {e}")
            continue
        width, height = image.size
        quarters = [
            image.crop((0, 0, width / 2, height / 2)),
            image.crop((width / 2, 0, width, height / 2)),
            image.crop((0, height / 2, width / 2, height)),
            image.crop((width / 2, height / 2, width, height)),
        ]
        for i, quarter in enumerate(quarters):
            quarter_filepath = os.path.join(
                "data/eval_on/quarter_images/", f"{assembly}_{page_str}_q{i}.png"
            )
            try:
                quarter.save(quarter_filepath)
            except Exception as e:
                print(f"Error saving quarter image {quarter_filepath}: {e}")
                continue
            for model in models:
                key = (model, assembly, page_str)
                if key not in tasks:
                    tasks[key] = []
                tasks[key].append(
                    asyncio.create_task(find_element_ids_on_page(quarter_filepath, model))
                )
    aggregated_results = {}
    for key, task_list in tasks.items():
        results = await asyncio.gather(*task_list)
        union_set = set()
        for res in results:
            union_set.update(res)
        aggregated_results[key] = union_set

    predicted_mapping = {"gemini-2.0-flash-001": {}, "gpt-4o": {}, "o1": {}}
    for (model, assembly, page_str), elem_ids in aggregated_results.items():
        for elem in elem_ids:
            key = (assembly, elem)
            if key not in predicted_mapping[model]:
                predicted_mapping[model][key] = set()
            predicted_mapping[model][key].add(page_str)

    hallucinations_g, missed_g, disagreements_g, common_count_g = compute_metrics(
        predicted_mapping["gemini-2.0-flash-001"], csv_labeling
    )
    hallucinations_4o, missed_4o, disagreements_4o, common_count_4o = compute_metrics(
        predicted_mapping["gpt-4o"], csv_labeling
    )
    hallucinations_o1, missed_o1, disagreements_o1, common_count_o1 = compute_metrics(
        predicted_mapping["o1"], csv_labeling
    )

    hallucination_rate_g = hallucinations_g / (
        len(predicted_mapping["gemini-2.0-flash-001"])
        if predicted_mapping["gemini-2.0-flash-001"]
        else 1
    )
    missed_rate_g = missed_g / (len(csv_labeling) if csv_labeling else 1)
    disagreement_rate_g = disagreements_g / (common_count_g if common_count_g else 1)

    hallucination_rate_4o = hallucinations_4o / (
        len(predicted_mapping["gpt-4o"]) if predicted_mapping["gpt-4o"] else 1
    )
    missed_rate_4o = missed_4o / (len(csv_labeling) if csv_labeling else 1)
    disagreement_rate_4o = disagreements_4o / (common_count_4o if common_count_4o else 1)

    hallucination_rate_o1 = hallucinations_o1 / (
        len(predicted_mapping["o1"]) if predicted_mapping["o1"] else 1
    )
    missed_rate_o1 = missed_o1 / (len(csv_labeling) if csv_labeling else 1)
    disagreement_rate_o1 = disagreements_o1 / (common_count_o1 if common_count_o1 else 1)

    print(f"Gemini Hallucination rate: {hallucination_rate_g:.2%}")
    print(f"Gemini Missed rate: {missed_rate_g:.2%}")
    print(f"Gemini Disagreement rate: {disagreement_rate_g:.2%}")
    print(f"gpt-4o Hallucination rate: {hallucination_rate_4o:.2%}")
    print(f"gpt-4o Missed rate: {missed_rate_4o:.2%}")
    print(f"gpt-4o Disagreement rate: {disagreement_rate_4o:.2%}")
    print(f"o1 Hallucination rate: {hallucination_rate_o1:.2%}")
    print(f"o1 Missed rate: {missed_rate_o1:.2%}")
    print(f"o1 Disagreement rate: {disagreement_rate_o1:.2%}")

    all_keys = (
        set(csv_labeling.keys())
        | set(predicted_mapping["gemini-2.0-flash-001"].keys())
        | set(predicted_mapping["gpt-4o"].keys())
        | set(predicted_mapping["o1"].keys())
    )
    rows = []
    for key in all_keys:
        assembly, elem = key
        labeled_page = csv_labeling.get(key, "N/A")
        pages_g = predicted_mapping["gemini-2.0-flash-001"].get(key, set())
        pages_4o = predicted_mapping["gpt-4o"].get(key, set())
        pages_o1 = predicted_mapping["o1"].get(key, set())
        quarter_page_str_g = ",".join(sorted(pages_g)) if pages_g else "N/A"
        quarter_page_str_4o = ",".join(sorted(pages_4o)) if pages_4o else "N/A"
        quarter_page_str_o1 = ",".join(sorted(pages_o1)) if pages_o1 else "N/A"
        rows.append(
            {
                "Assembly ID": assembly,
                "Element ID": elem,
                "Page ID OG": labeled_page,
                "Quarter Page ID Gemini": quarter_page_str_g,
                "Quarter Page ID 4o": quarter_page_str_4o,
                "Quarter Page ID o1": quarter_page_str_o1,
            }
        )

    output_df = pd.DataFrame(
        rows,
        columns=[
            "Assembly ID",
            "Element ID",
            "Page ID OG",
            "Quarter Page ID Gemini",
            "Quarter Page ID 4o",
            "Quarter Page ID o1",
        ],
    )

    output_csv_path = os.path.join("data", "labeled_by_quarters.csv")
    input_order_index = pd.MultiIndex.from_frame(df[["Assembly ID", "Element ID"]])
    output_df_index = pd.MultiIndex.from_frame(output_df[["Assembly ID", "Element ID"]])

    # 3. Reindex output_df to match the input order
    # First, temporarily set the MultiIndex on output_df
    output_df = output_df.set_index(["Assembly ID", "Element ID"])

    # Now reindex using the input order index
    output_df_reordered = output_df.reindex(input_order_index)

    # 4. Reset index to get back regular columns
    output_df_reordered = output_df_reordered.reset_index()

    output_csv_path = os.path.join("data", "llm_results", "labeled_by_quarters.csv")
    os.make_dirs(output_csv_path, exiss_ok=True)
    output_df_reordered.to_csv(output_csv_path, index=False)
    print(f"Output CSV written to {output_csv_path}")
