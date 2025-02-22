# %%
# Mappings from element_ids to which page their on
# This script to be run in repl so can confirm first
%load_ext autoreload
%autoreload 2
#%load_ext nb_black

import math
import re
import pandas as pd
import json
import os
import base64

#import litellm
import openai
import asyncio
import nest_asyncio

from src.image_cost import validate_cost

nest_asyncio.apply()

import concurrent.futures
custom_executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
loop = asyncio.get_event_loop()
loop.set_default_executor(custom_executor)


def load_answer_key(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # We'll add a Page column if it doesn't exist:
    if "Page" not in df.columns:
        df["Page"] = ""
    return df

def get_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    return image_base64

def find_element_ids_on_page(image_path: str) -> list:
    """
    Sends the image to GPT-4o (through some endpoint) to get a JSON list of element IDs.
    In practice, you'll likely need image + text prompt with instructions, or an OCR step.
    """
    # Example text prompt
    image_base64 = get_base64(image_path)

    prompt = f"""You are given an image with mechanical drawings.
Return a JSON array of all "Element IDs" visible, e.g. ["T1", "D1", "DF1"]. Go over the image slowly and carefully to return every last one.
Be sure to only respond with valid JSON within the <answer></answer> tags but you're allowed to think outside of those to make sure there's the highest accuracy"""
    response = openai.chat.completions.create(
        model="o1",
        max_completion_tokens=10000,
        #model="gpt-4o",
        #max_tokens=1000,
        messages=[{"role": "user",
                   "content":[{"type":"text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                         "detail": "high",
                         }
                     }],
        }],
    )
    reply_content = response.choices[0].message.content
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
        print(response)

    return []


async def async_find_element_ids_on_page_best(image_path: str) -> list:
    """
    Runs three asynchronous calls (using asyncio.to_thread) to get element IDs
    from the same image, then returns a consensus list containing only those
    element IDs that appear in at least two of the three calls.
    """
    best_of = 1
    min_n = 1 # math.ceil(best_of/2) # at least n to be included
    tasks = [asyncio.to_thread(find_element_ids_on_page, image_path) for _ in range(best_of)]
    results = await asyncio.gather(*tasks)
    counter = {}
    for result in results:
        for elem in result:
            counter[elem] = counter.get(elem, 0) + 1
    # Include only those element IDs returned at least twice.
    consensus = [elem for elem, count in counter.items() if count >= min_n]
    if best_of > 1:
        print(
        f"""Best-of-{best_of} Dropped IDs for {image_path}:
        {",".join([elem for elem, count in counter.items() if count < min_n])}"""
        )
    return consensus

# Asynchronous function to ask model O1 for a final correction.
async def async_compute_final_correction(assembly_id: int, original_labeling: dict, error_messages: str, image_paths: list) -> dict:
    functions = [
        {
            "name": "extract_element_ids",
            "description": "Extracts a list of element IDs from an image analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of element IDs extracted from the image."
                    }
                },
                "required": ["element_ids"]
            }
        }
    ]

    prompt = (
        f"You are provided with the following data for assembly {assembly_id}:\n\n"
        f"Original Labeling (mapping page -> element IDs):\n{json.dumps(original_labeling, indent=2)}\n\n"
        f"Error Messages:\n{error_messages}\n\n"
        f"Image file paths for the assembly:\n" + "\n".join(image_paths) + "\n\n"
        "Based on the above, please provide the corrected labeling as a JSON object mapping page numbers to lists of element IDs. "
        "Only include pages present in the images."
    )

    # Build the multimodal content payload.
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            *[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{get_base64(p)}",
                        "detail": "high"
                    }
                } for p in image_paths
            ]
        ]
    }]

    # Call the O1 model asynchronously using asyncio.to_thread.
    response = await asyncio.to_thread(
        openai.chat.completions.create,
        model="o1",
        max_completion_tokens=50000,
        messages=messages,
        functions=functions,
        function_call={"name": "extract_element_ids"},
    )

    # Extract reply content.
    try:
        message = response.choices[0].message
        # Use attribute access instead of .get()
        if hasattr(message, "function_call") and message.function_call is not None:
            reply_content = message.function_call.arguments or "{}"
        elif hasattr(message, "content") and message.content is not None:
            reply_content = message.content or "{}"
        else:
            reply_content = "{}"
    except (AttributeError, IndexError) as e:
        print(f"Error accessing the response message content: {e}")
        print(response)
        print(response.choices[0].message)
        return {}

    # Parse and return the corrected labeling.
    try:
        corrected_labeling = json.loads(reply_content)
        if isinstance(corrected_labeling, dict):
            return corrected_labeling
        print(f"Unexpected format from O1 correction: {reply_content}")
        return {}
    except Exception as e:
        print(f"Error parsing JSON from O1 correction: {e}")
        print(response)
        print(reply_content)
        return {}


async def update_csv_with_pages(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    1. Retrieves all file paths (with assembly id and page number) from the image_dir.
    2. Validates cost by passing all file paths at once.
    3. Processes all images concurrently (using best-of-3 sampling) and stores the results in a
       {(assembly_id, page): [elem_ids]} dictionary.
    4. Validates that element IDs are unique per page for a given assembly and that the set of IDs
       matches those in the CSV.
    5. Updates the "Page" column in the DataFrame accordingly.
    """
    # Step 1: Gather file paths and associated metadata.
    files_info = []
    for file_name in sorted(os.listdir(image_dir)):
        if not file_name.endswith(".png"):
            continue
        # Extract assembly ID using regex; e.g. "nist_ftc_06..."
        match = re.search(r"nist_ftc_(\d+)", file_name)
        if not match:
            continue
        assembly_id = int(match.group(1))
        # Extract page number: assume it's everything after '_pg' until ".png"
        page = int(file_name.split("_pg")[-1].replace(".png", ""))
        files_info.append((os.path.join(image_dir, file_name), assembly_id, page))

    print(files_info)

    # Step 2: Validate cost for all files at once.
    filepaths = [info[0] for info in files_info]
    #validate_cost(filepaths * 3, model="gpt-4o")  # since using best of 3
    validate_cost(filepaths, model="o1")  # since using best of 3

    # Step 3: Start all async calls concurrently with best-of-3 sampling.
    # or used previous cached
    if False:
        tasks = {}
        for filepath, assembly_id, page in files_info:
            key = (assembly_id, page)
            tasks[key] = asyncio.create_task(async_find_element_ids_on_page_best(filepath))
        results = await asyncio.gather(*tasks.values())
        # Build a dict mapping (assembly_id, page) -> consensus list of element IDs.
        results_dict = {key: result for key, result in zip(tasks.keys(), results)}
    else:
        # saved responses
        # 4o
        #old_results_dict = {(6, 1): ['STR5', 'T23', 'D14', 'D15', 'D13', 'D7', 'T11', 'T4', 'T5', 'DF4', 'D12-1', 'D12-2', 'T15', 'STR4', 'D8', 'D9-1', 'D12', 'T1', 'DF1', 'D2', 'DF6', 'T13', 'D11-1', 'T14', 'CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7'], (6, 2): ['T6', 'T7', 'DF5', 'D16', 'T16', 'D17', 'T17', 'T18', 'D1', 'T2', 'DF2', 'T3', 'DF3'], (6, 3): ['T1', 'D1', 'DF1', 'D3', 'DT1', 'D22', 'T21', 'DT3', 'DT4', 'J1', 'J2', 'D5', 'D6', 'T9', 'STR2', 'RLE3', 'RLE4', 'CS10', 'DT5', 'K1', 'D18', 'D19', 'T10', 'STR3', 'DT6', 'K2', 'CS11', 'D20', 'D21', 'T20', 'D4', 'RLE1', 'CS8', 'G1', 'H1', 'RLE2', 'CS9', 'DT2', 'CS1', 'RLE5', 'RLE6'], (7, 1): ['STR8', 'T19', 'D1', 'D2', 'DF2', 'D3', 'DF3', 'D4', 'D5', 'T6', 'D7', 'D8', 'T7', 'F1.2', 'DF1', 'T1', 'STR1', 'LDN2', 'LDN1', 'CSI'], (7, 2): ['T1', 'D1', 'DF1', 'D9', 'D10', 'D11', 'T4', 'STR2', 'DF4', 'T8', 'T9', 'T10', 'VS11', 'CS3', 'D12', 'D13', 'T12', 'T15', 'D14', 'T13', 'D15', 'T14'], (7, 3): ['T15', 'T16', 'T18', 'DF5', 'STR3', 'STR4', 'STR7', 'T5'], (7, 4): ['D17', 'T16', 'T17', 'STR5', 'STR6'], (8, 1): ['STR6', 'T28', 'T13', 'T1', 'T2', 'T14', 'T15', 'D9', 'D1', 'D2', 'D3', 'Df1'], (8, 2): ['T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'E', 'F', 'D', 'DF3', 'DF4', 'DF5', 'DF6', 'D3', 'T7', 'T5', 'T6', 'RLE1', 'RLE2', 'L1', 'L2', 'LDN1', 'LDN2', '5TR5', 'CH1', 'CS1', 'CS2', 'CS3'], (8, 3): ['T25', 'CS2', 'T24', 'D4', 'T8', 'DT7', 'D10'], (8, 4): ['T26', 'T27', 'D5', 'D6', 'D7', 'D8', 'T3', 'DF8', 'STRL', 'DF6', 'T10', 'STR2', 'DF9', 'T12', 'DF11', 'CS5', 'CS2'], (9, 1): ['STR6', 'D1', 'D2', 'D8', 'D9', 'D10', 'D11', 'DF1', 'DF2', 'DF3', 'T1', 'T2', 'T3', 'T12', 'T13', 'T14'], (9, 2): ['CS1', 'CS2', 'CS3', 'D5', 'D6', 'D12', 'D13', 'D14', 'D15', 'D16', 'T7', 'T15', 'T16', 'T17', 'VS1', 'STR1', 'STR2', 'STR3'], (9, 3): ['T19', 'T22', 'T21', 'T11', 'D7', 'D6', 'DF7', 'T20', 'D17', 'T10', 'T9', 'T8', 'T7', 'DF8', 'D8', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'CS4', 'CS1'], (9, 4): ['D3', 'T3', 'T5', 'DF4', 'D4', 'D5', 'DF5', 'T6', 'D20', 'T27', 'STR4', 'D21', 'T26', 'STR5', 'D18', 'T23', 'D19', 'T24', 'T25'], (10, 1): ['STR11', 'T37', 'D2', 'T3', 'DF3', 'T16', 'STR2', 'D8', 'FN4', 'T15', 'T1', 'DF1', 'CS1', 'D1', 'T2', 'DF2', 'D7', 'T14'], (10, 2): ['T1', 'T4', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'T21', 'T22', 'T23', 'T25', 'DFA', 'STR1', 'STR2', 'STR3', 'STR4', 'STR5', 'STR6', 'LDN1', 'LDN2', 'LDN3', 'CS1', 'CS3', 'F65', 'RLE1', 'Z1'], (10, 3): ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'DF1', 'DF2', 'FNS', 'FN1', 'FN2', 'FN3', 'T1', 'T2', 'T3', 'T6', 'T7', 'T26', 'T27', 'T28', 'T29', 'T30'], (10, 4): ['MDT1', 'T11', 'DF8', 'L1', 'P2', 'CS11', 'SG2', 'DT1', 'K1', 'MDT2', 'D23', 'T32', 'D24', 'T33', 'D25', 'T34', 'CS4', 'CS5', 'CS6', 'DT2', 'K2', 'L2'], (10, 5): ['D26', 'T35', 'T36', 'CS1', 'Z'], (11, 1): ['DF2', 'B', 'VS1', 'XCS1', 'D2', 'D1', 'T2', 'T3', 'D3', 'D4'], (11, 2): ['T4', 'D6', 'D5', 'T1', 'DF1']}
        #results_dict = {(6, 1): ['STR5', 'T23', 'Z1', 'CS2', 'CS1', 'D14', 'D15', 'D13', 'D7', 'T11', 'T4', 'T5', 'DF4', 'D10', 'T13', 'D11-2', 'T14', 'CS6', 'D8', 'D9-1', 'T12', 'D9-2', 'T1', 'DF1', 'D2', 'D12-1', 'T15', 'D12-2', 'STR4', 'CS4', 'CS5', 'CS3', 'DF6', 'STR1', 'D1-2', 'STR3', 'D11-1', 'CS7', 'PF6', 'D11', 'D1'], (6, 2): ['T6', 'T7', 'DF5', 'D16', 'T18', 'T17', 'D17', 'D1', 'T2', 'DF2', 'T3', 'DF3', 'T1', 'T16', 'D2', 'D3', 'DF1'], (6, 3): ['D22', 'T21', 'DT3', 'T9', 'STR2', 'RLE3', 'RLE4', 'CS10', 'DT5', 'K1', 'RLE5', 'CS11', 'DT6', 'K2', 'T10', 'STR3', 'D18', 'D19', 'T19', 'DT1', 'G1', 'D3', 'DT2', 'H1', 'D6', 'D20', 'D21', 'T20', 'CS8', 'RLE1', 'D4', 'CS9', 'RLE2', 'D1', 'D5', 'D9', 'D10', 'DT4', 'J1', 'J2', 'RLE6', 'T22'], (7, 1): ['STR8', 'D1', 'D2', 'D3', 'DF1', 'D5', 'D6', 'D7', 'D8', 'T1', 'T6', 'T7', 'T19', 'F1.2', 'LDN1', 'LDN2', 'DF3', 'CSI', 'F1.1', 'D4', 'D9', 'DF2', 'STR1', 'C', 'D'], (7, 2): ['T1', 'D1', 'T4', 'STR2', 'DF4', 'T8', 'T9', 'T10', 'VS11', 'D12', 'D13', 'T12', 'CS3', 'T13', 'D14', 'T14', 'D15', 'D9', 'D10', 'D11', 'T11', 'VS1'], (7, 3): ['D16', 'T15', 'T18', 'STR7', 'T5', 'STR3', 'STR4', 'DF5', 'A', 'B', 'C', 'E', 'C3', 'T1'], (7, 4): ['D17', 'STR5', 'T16', 'STR6', 'T17', 'F6'], (8, 1): ['STR6', 'T28', 'D9', 'T13', 'T1', 'T2', 'Df1', 'T14', 'T15', 'D1', 'T4', 'Df3', 'D2', 'CS1', 'CS2', 'Df2', 'DF2', 'D3', 'DF1'], (8, 2): ['T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'D1', 'D3', 'T7', 'T5', 'T6', 'L1', 'L2', 'L3', 'DF5', 'DF6', 'RLE1', 'RLE2', 'LDN1', 'LDN2', 'CH1', 'CS1', 'CS2', '5TRS', 'ST2', 'ST3', 'D4', 'D5', 'DF3', 'DF4', 'STRS', 'E', 'F', 'T1', 'DF1', 'CS3', '5TR5'], (8, 3): ['T25', 'CS2', 'T24', 'D4', 'T8', 'DT7', 'D10', 'S4'], (8, 4): ['T2', 'D5', 'T3', 'DF8', 'STRL', 'D6', 'T10', 'DF9', 'DF10', 'DF11', 'T12', 'STRM', 'D8', 'T11', 'D7', 'T27', 'T26', 'CS2', 'CS6', 'STR2', 'STR4', 'STR1', 'T6', 'T7', 'STR6', 'CS5', 'STRL1'], (9, 1): ['STR6', 'D2', 'T3', 'DF3', 'T1', 'DF1', 'D11', 'T14', 'D10', 'T13', 'D9', 'T12', 'D8', 'CS1X', 'D1', 'T2', 'DF2', 'T22', 'T29'], (9, 2): ['D5', 'T7', 'D16', 'D15', 'T18', 'D13', 'T17', 'T12', 'T16', 'D14', 'D3', 'D6', 'D12', 'T15', 'VS1', 'CS1', 'CS3', 'STR2', 'STR1', 'STR3', 'T13'], (9, 3): ['T22', 'D15', 'D6', 'D17', 'D18', 'D7', 'T8', 'D16', 'T19', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'T11', 'D8', 'T9', 'DF7', 'DF8', 'T10', 'CS4', 'CS1', 'T20', 'T21', 'T17', 'DT7', 'DT8', 'T12', 'T18'], (9, 4): ['D3', 'T3', 'DF4', 'D4', 'T6', 'DF5', 'DFS', 'D20', 'T27', 'STR4', 'D19', 'T24', 'T25', 'D21', 'T28', 'STR5', 'DF1', 'T2', 'S1', 'T12', 'T26', 'RL6', 'T5', 'D18', 'T23', 'RLE6', 'DF7', 'DF6', 'RL25'], (10, 1): ['STR11', 'T37', 'T16', 'T1', 'DF1', 'D1', 'D2', 'D3', 'DF3', 'D8', 'FN4', 'T15', 'CS1', 'T14', 'D7', 'DF2', 'STR2', 'T2', 'T3'], (10, 2): ['T1', 'D1', 'DF1', 'D14', 'D15', 'T19', 'T20', 'STR4', 'STR5', 'RLE2', 'T21', 'T22', 'DF4', 'STR2', 'STR3', 'T18', 'D16', 'T25', 'LDN1', 'LDN2', 'CS3', 'RLE1', 'CS2', 'T4', 'F65', 'T23', 'D17', 'D12', 'D13', 'D11', 'D9', 'D10', 'D18', 'LDN3', 'T2', 'T3', 'T5', 'T6', 'T7', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'STR1', 'STR6', 'DFA', 'OCS1', 'P27', 'T24', 'Z1'], (10, 3): ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'FNS', 'FN1', 'FN2', 'FN3', 'STR7', 'STR8', 'STR9', 'STR10', 'T6', 'T7', 'T10', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'Z1', 'CS1', 'CS5', 'DN4', 'F81', 'CS2', 'LDN4', 'DF5', 'DF6', 'LN4', 'CS3', 'D10', 'FN4'], (10, 4): ['MDT1', 'L1', 'T11', 'DF8', 'H', 'CS11', 'SG2', 'K1', 'DT1', 'PZ', 'CS12', 'MDT2', 'L2', 'D23', 'T32', 'D24', 'T33', 'D25', 'T34', 'CS5', 'CS4', 'CS6', 'K2', 'DT2', 'CS7', 'CS1', 'CS2'], (10, 5): ['D26', 'T35', 'T36', 'CS1', 'Z1'], (11, 1): ['CS1', 'VSL1', 'B', 'D1', 'D2', 'DF2', 'T2', 'T3', 'D4', 'D3', 'VS1', 'STR1'], (11, 2): ['T4', 'Z', 'CS1', 'D6', 'D5', 'T1', 'DF1', 'A']}
        #results_dict = {k: set(v + old_results_dict[k]) for k,v in results_dict.items()}
        # o1
        # {(6, 1): ['T1', 'T4', 'T5', 'T11', 'T12', 'T13', 'T14', 'T15', 'T23', 'D7', 'D8', 'D9-1', 'D9-2', 'D10', 'D11-1', 'D11-2', 'D12-1', 'D12-2', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'DF6', 'CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'STR1', 'STR4', 'STR5', 'Z1'], (6, 2): ['D1', 'D16', 'D17', 'DF2', 'DF3', 'DF5', 'T2', 'T3', 'T6', 'T7', 'T16', 'T17', 'T18'], (6, 3): ['CS1', 'D3', 'D4', 'D5', 'D6', 'D18', 'D19', 'D20', 'D21', 'D22', 'T9', 'T10', 'T19', 'T20', 'T21', 'T22', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5', 'DT6', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'RLE5', 'RLE6', 'CS8', 'CS9', 'CS10', 'CS11', 'G1', 'H1', 'J1', 'J2', 'K1', 'K2', 'STR2', 'STR3'], (7, 1): ['D1', 'D2', 'D3', 'D5', 'D7', 'D8', 'DF1', 'DF2', 'F1.1', 'F1.2', 'LDN1', 'LDN2', 'STR1', 'STR8', 'T1', 'T6', 'T7', 'T19'], (7, 2): ['CS3', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'DF4', 'STR2', 'T4', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'VS1'], (7, 3): ['D16', 'T15', 'T5', 'T18', 'STR3', 'STR4', 'STR7', 'DF5', 'C3'], (7, 4): ['D17', 'STR5', 'T16', 'STR6', 'T17'], (8, 1): ['T1', 'T2', 'DF1', 'T13', 'T14', 'T15', 'D9', 'T28', 'STR6', 'D2', 'DF2', 'CS1', 'CS2'], (8, 2): ['T5', 'T6', 'T7', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'D3', 'DF4', 'DF5', 'DF6', 'RLE1', 'RLE2', 'LDN1', 'LDN2', 'CH1', 'L1', 'L2', 'CS1', 'CS2'], (8, 3): ['T24', 'T25', 'D4', 'T8', 'DF7', 'D10', 'S4', 'CS2'], (8, 4): ['T3', 'T10', 'T12', 'T26', 'T27', 'D5', 'D6', 'D7', 'D8', 'DF3', 'DF8', 'DF11', 'STR1', 'STR2', 'CS2', 'CS5', 'CS6'], (9, 1): ['CS1', 'D1', 'D2', 'D8', 'D9', 'D10', 'D11', 'DF1', 'DF2', 'DF3', 'T1', 'T2', 'T3', 'T11', 'T12', 'T13', 'T14'], (9, 2): ['A', 'B', 'C', 'F', 'CS1', 'CS3', 'D5', 'D6', 'D12', 'D14', 'D15', 'D16', 'STR1', 'STR3', 'T7', 'T15', 'T16', 'T17', 'T18', 'TPR2', 'VS1'], (9, 3): ['CS1', 'CS4', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'D6', 'T8', 'T9', 'DF7', 'D7', 'T10', 'DF8', 'T11', 'T17', 'T19', 'T20', 'T21', 'T22'], (9, 4): ['D3', 'D4', 'D18', 'D19', 'D20', 'D21', 'DF4', 'DF5', 'RLE5', 'RLE6', 'S1', 'S2', 'S3', 'S4', 'STR4', 'T3', 'T5', 'T6', 'T23', 'T24', 'T25', 'T26', 'T27'], (10, 1): ['STR11', 'T37', 'T1', 'DF1', 'T15', 'D8', 'FN4', 'T16', 'STR2', 'T14', 'D7', 'CS1'], (10, 2): ['CS2', 'CS3', 'D6', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'DFA', 'LDN1', 'LDN2', 'LDN3', 'RLE1', 'STR1', 'STR3', 'STR4', 'STR5', 'STR6', 'T4', 'T12', 'T13', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'Z1'], (10, 3): ['CS1', 'CS3', 'D3', 'D4', 'D5', 'D6', 'D7', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'DF5', 'DF6', 'DF7', 'FN1', 'FN2', 'FN3', 'FN4', 'FN5', 'F81', 'FNS', 'LDN4', 'STR7', 'STR8', 'STR9', 'STR10', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'Z1'], (10, 4): ['CS1z', 'CS2', 'CS4', 'CS5', 'CS6', 'D23', 'D24', 'D25', 'DF8', 'DT1', 'DT2', 'H', 'K1', 'K2', 'L1', 'L2', 'MDT1', 'MDT2', 'SG2', 'T11', 'T32', 'T33', 'T34'], (10, 5): ['A', 'B', 'C', 'D26', 'T35', 'T36', 'CS1', 'Z1'], (11, 1): ['A', 'B', 'CS1', 'D1', 'D2', 'D3', 'D4', 'DF2', 'E', 'STR1', 'T2', 'T3', 'VSL1'], (11, 2): ['A', 'B', 'CS1', 'D5', 'D6', 'DF1', 'T1', 'T4']}
        # Gemini evaluating previous to standardize on results
        # results_dict = {(6, 1): ['D8',   'D9-1',   'D11-1',   'D12-2',   'D13',   'D14',   'D15',   'DF1',   'DF4',   'DF6',   'STR1',   'STR4',   'STR5',   'T1',   'T4',   'T5',   'T11',   'T12',   'T13',   'T14',   'T15',   'T23'],  (6, 2): ['D1',   'D16',   'D17',   'DF2',   'DF3',   'DF5',   'T2',   'T3',   'T6',   'T7',   'T16',   'T17',   'T18'],  (6, 3): ['D3',   'D4',   'D5',   'D6',   'D10',   'D18',   'D19',   'D20',   'D21',   'D22',   'DT1',   'DT2',   'DT3',   'DT4',   'DT5',   'DT6',   'G1',   'H1',   'J1',   'J2',   'K1',   'K2',   'RLE1',   'RLE2',   'RLE3',   'RLE4',   'RLE5',   'RLE6',   'STR2',   'STR3',   'T9',   'T10',   'T19',   'T20',   'T21',   'T22'],  (7, 1): ['D1',   'D2',   'D3',   'D5',   'D7',   'D8',   'DF1',   'DF2',   'DF3',   'F1.2',   'LDN1',   'LDN2',   'STR1',   'STR8',   'T1',   'T6',   'T7',   'T19'],  (7, 2): ['D9',   'D10',   'D11',   'D12',   'D13',   'D14',   'D15',   'DF1',   'DF4',   'STR2',   'T1',   'T4',   'T8',   'T9',   'T10',   'T11',   'T12',   'T13',   'T14'],  (7, 3): ['DF5', 'STR3', 'STR4', 'STR7', 'T5', 'T15', 'T18'],  (7, 4): ['D17', 'STR5', 'STR6', 'T16', 'T17'],  (8, 1): ['D2',   'D3',   'D9',   'DF1',   'DF2',   'STR6',   'T1',   'T2',   'T13',   'T14',   'T15',   'T28'],  (8, 2): ['D1',   'D4',   'D5',   'DF3',   'DF4',   'DF5',   'DF6',   'E',   'F',   'L1',   'L2',   'LDN1',   'LDN2',   'RLE1',   'RLE2',   'T5',   'T6',   'T7',   'T16',   'T17',   'T18',   'T19',   'T20',   'T21',   'T22',   'T23',   'CH1'],  (8, 3): ['D4', 'D10', 'DT7', 'T8', 'T24', 'T25'],  (8, 4): ['D5',   'D6',   'D7',   'D8',   'DF6',   'DF8',   'DF9',   'DF11',   'STR1',   'STR2',   'T2',   'T3',   'T6',   'T10',   'T12',   'T26',   'T27'],  (9, 1): ['D1',   'D2',   'D8',   'D9',   'D10',   'D11',   'DF1',   'DF2',   'DF3',   'STR6',   'T1',   'T2',   'T3',   'T12',   'T13',   'T14'],  (9, 2): ['D3',   'D5',   'D6',   'D12',   'D14',   'D15',   'D16',   'STR1',   'STR2',   'STR3',   'T7',   'T15',   'T16',   'T17',   'T18'],  (9, 3): ['D6',   'D7',   'D8',   'D15',   'D17',   'DF7',   'DF8',   'DT7',   'RLE1',   'RLE2',   'RLE3',   'RLE4',   'T8',   'T9',   'T10',   'T11',   'T19',   'T20',   'T21',   'T22'],  (9, 4): ['D3',   'D4',   'D5',   'D18',   'D19',   'D20',   'D21',   'DF1',   'DF4',   'DF5',   'DF6',   'STR4',   'STR5',   'T2',   'T3',   'T5',   'T6',   'T12',   'T23',   'T24',   'T25',   'T26',   'T27',   'T28'],  (10, 1): ['D1',   'D2',   'D3',   'D7',   'D8',   'DF1',   'DF2',   'DF3',   'FN4',   'STR2',   'T1',   'T2',   'T3',   'T14',   'T15',   'T16',   'T37'],  (10, 2): ['D1',   'D9',   'D10',   'D11',   'D12',   'D13',   'D14',   'D15',   'D16',   'D17',   'D18',   'D19',   'D20',   'DF1',   'DF4',   'DFA',   'F65',   'LDN1',   'LDN2',   'LDN3',   'RLE1',   'STR1',   'STR2',   'STR3',   'STR4',   'STR5',   'STR6',   'T1',   'T4',   'T5',   'T6',   'T7',   'T8',   'T9',   'T10',   'T11',   'T12',   'T13',   'T14',   'T15',   'T16',   'T17',   'T18',   'T19',   'T20',   'T21',   'T22',   'T23',   'T24',   'T25'],  (10, 3): ['D1',   'D2',   'D3',   'D4',   'D5',   'D6',   'D7',   'D17',   'D18',   'D19',   'D20',   'D21',   'D22',   'DF1',   'DF2',   'DF5',   'DF6',   'FN1',   'FN2',   'FN3',   'FN4',   'FNS',   'T1',   'T2',   'T3',   'T6',   'T7',   'T8',   'T9',   'T10',   'T26',   'T27',   'T28',   'T29',   'T30'],  (10, 4): ['D23',   'D24',   'D25',   'DF8',   'DT1',   'DT2',   'K1',   'K2',   'L1',   'L2',   'MDT1',   'MDT2',   'SG2',   'T11',   'T32',   'T33',   'T34'],  (10, 5): ['D26', 'T35', 'T36', 'Z1'],  (11, 1): ['D1', 'D2', 'D3', 'D4', 'DF2', 'T2', 'T3'],  (11, 2): ['D5', 'D6', 'DF1', 'T1', 'T4']}
        # Gemini Try 2
        #results_dict={(6, 1): ['D8', 'D9-1', 'D11-1', 'D12-2', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'DF6', 'STR1', 'STR4', 'T1', 'T4', 'T5', 'T11', 'T12', 'T13', 'T14', 'T15', 'T23'],  (6, 2): ['D1', 'D16', 'D17', 'DF2', 'DF3', 'DF5', 'T2', 'T3', 'T6', 'T7', 'T16', 'T17', 'T18'],  (6, 3): ['D3', 'D4', 'D5', 'D6', 'D10','D18', 'D19', 'D20', 'D21', 'D22', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5', 'DT6', 'G1', 'H1', 'J1', 'J2', 'K1', 'K2', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'RLE5', 'RLE6', 'STR2', 'STR3', 'T9', 'T10', 'T19', 'T20', 'T21', 'T22'],  (7, 1): ['D1', 'D2', 'D3', 'D5', 'D7', 'D8', 'DF1', 'DF2',  'LDN1', 'LDN2', 'STR1',  'T1', 'T6', 'T7', 'T19'],  (7, 2): ['D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'STR2', 'T1', 'T4', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14'],  (7, 3): ['DF5', 'STR3', 'STR4', 'STR7', 'T5', 'T15','T18'],  (7, 4): ['D17', 'STR5', 'STR6', 'T16', 'T17'],  (8, 1): ['D2', 'D3', 'D9', 'DF1', 'DF2',  'T1', 'T2', 'T13', 'T14', 'T15', 'T28'],  (8, 2): ['D1','DF3', 'DF4', 'DF5', 'DF6',  'LDN1', 'LDN2', 'RLE1', 'RLE2', 'T5', 'T6', 'T7', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'CH1'],  (8, 3): ['D4', 'D10',  'T8', 'T24', 'T25'],  (8, 4): ['D5', 'D6', 'D7', 'D8', 'DF6', 'DF8', 'DF9', 'DF11', 'STR1', 'STR2', 'T3', 'T10', 'T12', 'T26', 'T27'],  (9, 1): ['D1', 'D2', 'D8', 'D9', 'D10', 'D11', 'DF1', 'DF2', 'DF3',  'T1', 'T2', 'T3', 'T12', 'T13', 'T14'],  (9, 2): ['D3', 'D5', 'D6', 'D12', 'D14', 'D15', 'D16',  'STR1', 'STR2', 'STR3', 'T7', 'T15', 'T16', 'T17', 'T18'],  (9, 3): ['D6', 'D7', 'D8', 'D15', 'D17','DF7', 'DF8',  'RLE1', 'RLE2', 'RLE3', 'RLE4', 'T8', 'T9', 'T10', 'T11', 'T19', 'T20', 'T21', 'T22'],  (9, 4): ['D3', 'D4', 'D5', 'D18', 'D19', 'D20', 'D21', 'DF1', 'DF4', 'DF5', 'DF6', 'STR4', 'STR5', 'T2', 'T3', 'T5', 'T6', 'T12', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28'],  (10, 1): ['D1', 'D2', 'D3',  'D8', 'DF1', 'DF2', 'DF3', 'FN4', 'STR2', 'T1', 'T2', 'T3', 'T14', 'T15', 'T16', 'T37'],  (10, 2): ['D1', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',  'DF1', 'DF4', 'DFA', 'F65', 'LDN1', 'LDN2', 'LDN3', 'RLE1', 'STR1', 'STR3', 'STR4', 'STR5', 'STR6', 'T1', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T27'],  (10, 3): ['D1', 'D2', 'D3', 'D4', 'D5', 'D6',  'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'DF1', 'DF2', 'DF5', 'DF6', 'FN1', 'FN2', 'FN3', 'FN4',  'FNS', 'T1', 'T2', 'T3', 'T6', 'T7', 'T8', 'T9', 'T10', 'T26', 'T27', 'T28', 'T29', 'T30'],  (10, 4): ['D23', 'D24', 'D25', 'DF8', 'DT1', 'DT2', 'K1', 'K2', 'L1', 'L2', 'MDT1', 'MDT2', 'SG2', 'T11', 'T32', 'T33', 'T34'],  (10, 5): ['D26', 'T35', 'T36'],  (11, 1): ['D1', 'D2', 'D3', 'D4', 'DF2', 'T2', 'T3'],  (11, 2): ['D5', 'D6', 'DF1', 'T1', 'T4']}
        # Try 3 similar to 2
        # results_dict = {(6, 1): ['D8', 'D9-1', 'D11-1', 'D12-2', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'DF6', 'STR1', 'STR4', 'T1', 'T4', 'T5', 'T11', 'T12', 'T13', 'T14', 'T15', 'T23'],  (6, 2): ['D1', 'D16', 'D17', 'DF2', 'DF3', 'DF5', 'T2', 'T3', 'T6', 'T7', 'T16', 'T17', 'T18'],  (6, 3): ['D3', 'D4', 'D5', 'D6', 'D10','D18', 'D19', 'D20', 'D21', 'D22', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5', 'DT6', 'G1', 'H1', 'J1', 'J2', 'K1', 'K2', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'RLE5', 'RLE6', 'STR2', 'STR3', 'T9', 'T10', 'T19', 'T20', 'T21', 'T22'],  (7, 1): ['D1', 'D2', 'D3', 'D5', 'D7', 'D8', 'DF1', 'DF2',  'LDN1', 'LDN2', 'STR1',  'T1', 'T6', 'T7', 'T19'],  (7, 2): ['D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'STR2', 'T1', 'T4', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14'],  (7, 3): ['DF5', 'STR3', 'STR4', 'STR7', 'T5', 'T15','T18'],  (7, 4): ['D17', 'STR5', 'STR6', 'T16', 'T17'],  (8, 1): ['D2', 'D3', 'D9', 'DF1', 'DF2',  'T1', 'T2', 'T13', 'T14', 'T15', 'T28'],  (8, 2): ['D1','DF3', 'DF4', 'DF5', 'DF6',  'LDN1', 'LDN2', 'RLE1', 'RLE2', 'T5', 'T6', 'T7', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'CH1'],  (8, 3): ['D4', 'D10',  'T8', 'T24', 'T25'],  (8, 4): ['D5', 'D6', 'D7', 'D8', 'DF6', 'DF8', 'DF9', 'DF11', 'STR1', 'STR2', 'T3', 'T10', 'T12', 'T26', 'T27'],  (9, 1): ['D1', 'D2', 'D8', 'D9', 'D10', 'D11', 'DF1', 'DF2', 'DF3',  'T1', 'T2', 'T3', 'T12', 'T13', 'T14'],  (9, 2): ['D3', 'D5', 'D6', 'D12', 'D14', 'D15', 'D16',  'STR1', 'STR2', 'STR3', 'T7', 'T15', 'T16', 'T17', 'T18'],  (9, 3): ['D6', 'D7', 'D8', 'D15', 'D17','DF7', 'DF8',  'RLE1', 'RLE2', 'RLE3', 'RLE4', 'T8', 'T9', 'T10', 'T11', 'T19', 'T20', 'T21', 'T22'],  (9, 4): ['D3', 'D4', 'D5', 'D18', 'D19', 'D20', 'D21', 'DF1', 'DF4', 'DF5', 'DF6', 'STR4', 'STR5', 'T2', 'T3', 'T5', 'T6', 'T12', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28'],  (10, 1): ['D1', 'D2', 'D3',  'D8', 'DF1', 'DF2', 'DF3', 'FN4', 'STR2', 'T1', 'T2', 'T3', 'T14', 'T15', 'T16', 'T37'],  (10, 2): ['D1', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',  'DF1', 'DF4', 'DFA', 'F65', 'LDN1', 'LDN2', 'LDN3', 'RLE1', 'STR1', 'STR3', 'STR4', 'STR5', 'STR6', 'T1', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25'],  (10, 3): ['D1', 'D2', 'D3', 'D4', 'D5', 'D6',  'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'DF1', 'DF2', 'DF5', 'DF6', 'FN1', 'FN2', 'FN3', 'FN4',  'FNS', 'T1', 'T2', 'T3', 'T6', 'T7', 'T8', 'T9', 'T10', 'T26', 'T27', 'T28', 'T29', 'T30'],  (10, 4): ['D23', 'D24', 'D25', 'DF8', 'DT1', 'DT2', 'K1', 'K2', 'L1', 'L2', 'MDT1', 'MDT2', 'SG2', 'T11', 'T32', 'T33', 'T34'],  (10, 5): ['D26', 'T35', 'T36'],  (11, 1): ['D1', 'D2', 'D3', 'D4', 'DF2', 'T2', 'T3'],  (11, 2): ['D5', 'D6', 'DF1', 'T1', 'T4']}
        # Gemini after taking O1's attempt at fixing
        results_dict = {     (6, 1): ['D7', 'D8', 'D9-1', 'D9-2', 'D11-1', 'D11-2', 'D12-1', 'D12-2', 'D13', 'D14', 'D15', 'DF1', 'DF4', 'DF6', 'STR1', 'STR4', 'T1', 'T4', 'T5', 'T8', 'T11', 'T12', 'T13', 'T14', 'T15', 'T23'],     (6, 2): ['D1', 'D2', 'D16', 'D17', 'DF2', 'DF3', 'DF5', 'T2', 'T3', 'T6', 'T7', 'T16', 'T17', 'T18'],     (6, 3): ['D3', 'D4', 'D5', 'D6', 'D10', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5', 'DT6', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'RLE5', 'RLE6', 'STR2', 'STR3', 'T9', 'T10', 'T19', 'T20', 'T21', 'T22'],     (7, 1): ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'DF1', 'DF2', 'DF3', 'LDN1', 'LDN2', 'STR1', 'T1', 'T2', 'T3', 'T6', 'T7', 'T19'],     (7, 2): ['D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'DF4', 'STR2', 'T4', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14'],     (7, 3): ['D16', 'DF5', 'STR3', 'STR4', 'STR7', 'T5', 'T15', 'T18'],     (7, 4): ['D17', 'STR5', 'STR6', 'T16', 'T17'],     (8, 1): ['D2', 'D3', 'D9', 'DF1', 'DF2', 'T1', 'T2', 'T13', 'T14', 'T15', 'T28'],     (8, 2): ['D1', 'DF3', 'DF4', 'DF5', 'DF6', 'LDN1', 'LDN2', 'RLE1', 'RLE2', 'T5', 'T6', 'T7', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'CH1'],     (8, 3): ['D4', 'D10', 'T8', 'T24', 'T25'],     (8, 4): ['D5', 'D6', 'D7', 'D8', 'DF8', 'DF9', 'DF11', 'STR1', 'STR2', 'T3', 'T10', 'T12', 'T26', 'T27'],     (9, 1): ['D1', 'D2', 'D9', 'D10', 'D11', 'DF1', 'DF2', 'DF3', 'T1', 'T2', 'T3', 'T12', 'T13', 'T14'],     (9, 2): ['D13', 'D5', 'D12', 'D14', 'D16', 'STR1', 'STR2', 'STR3', 'T7', 'T15', 'T16', 'T17', 'T18'],     (9, 3): ['D6', 'D7', 'D8', 'D15', 'D17', 'DF7', 'DF8', 'RLE1', 'RLE2', 'RLE3', 'RLE4', 'RLE5', 'RLE6', 'T8', 'T9', 'T10', 'T11', 'T19', 'T20', 'T21', 'T22'],     (9, 4): ['D4', 'D18', 'D19', 'D20', 'D21', 'DF4', 'DF5', 'DF6', 'STR4', 'STR5', 'T4', 'T5', 'T6', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29'],     (10, 1): ['D1', 'D2', 'D3', 'D7', 'D8', 'DF1', 'DF2', 'DF3', 'FN4', 'STR2', 'T1', 'T2', 'T3', 'T14', 'T15', 'T16', 'T37'],     (10, 2): ['D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'DF4', 'LDN1', 'LDN2', 'LDN3', 'LDN4', 'RLE1', 'RLE2', 'STR1', 'STR3', 'STR4', 'STR5', 'STR6', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25'],     (10, 3): ['D4', 'D5', 'D6', 'D21', 'D22', 'DF5', 'DF6', 'DF7', 'DF9', 'FN1', 'FN2', 'FN3', 'FN5', 'STR7', 'STR8', 'STR9', 'STR10', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31'],     (10, 4): ['D23', 'D24', 'D25', 'DF8', 'DT1', 'DT2', 'MDT1', 'MDT2', 'SG1', 'SG2', 'T32', 'T33', 'T34'], (10, 5): ['D26', 'T35', 'T36'],  (11, 1): ['D1', 'D2', 'D3', 'D4', 'DF2', 'T2', 'T3'],  (11, 2): ['D5', 'D6', 'DF1', 'T1', 'T4'] }
    print(results_dict)

    # Group results by assembly.
    assembly_pages = {}
    for (assembly_id, page), elem_ids in results_dict.items():
        assembly_pages.setdefault(assembly_id, {})[page] = elem_ids

    # Step 4: Validate results.
    errors_by_assembly = {}  # assembly_id -> error messages string
    for assembly_id, pages_dict in assembly_pages.items():
        seen = {}
        error_msgs = []
        # Check for duplicate element IDs across pages.
        for page, elem_ids in pages_dict.items():
            for elem in elem_ids:
                if elem in seen and seen[elem] != page:
                    error_msgs.append(f"Element ID '{elem}' appears on both page {seen[elem]} and page {page}.")
                else:
                    seen[elem] = page

        # Compare union of found IDs with expected CSV labeling.
        if "Assembly ID" in df.columns:
            expected_ids = set(df.loc[df["Assembly ID"] == assembly_id, "Element ID"])
        else:
            expected_ids = set(df["Element ID"])
        found_ids = set(seen.keys())
        missing = expected_ids - found_ids
        extra = [i for i in (found_ids - expected_ids) if 'CS' not in i and 'VS' not in i]
        if missing:
            error_msgs.append(f"Missing element IDs: {missing}")
        if extra:
            error_msgs.append(f"Extra element IDs: {extra}")
        if error_msgs:
            errors_by_assembly[assembly_id] = "\n".join(error_msgs)
            print(f"Assembly {assembly_id} discrepancies:\n{errors_by_assembly[assembly_id]}\n")
        else:
            errors_by_assembly[assembly_id] = ""  # No errors

    # Step 5: For assemblies with discrepancies, call O1 to compute final correction concurrently.
    if False:
        correction_tasks = {
            assembly_id: asyncio.create_task(
                async_compute_final_correction(
                    assembly_id,
                    assembly_pages[assembly_id],  # Original labeling for this assembly.
                    error_msg,
                    [info[0] for info in files_info if info[1] == assembly_id]  # All image paths for this assembly.
                )
            )
            for assembly_id, error_msg in errors_by_assembly.items()
            if error_msg
        }

        # Await all tasks concurrently
        correction_results = await asyncio.gather(*correction_tasks.values(), return_exceptions=True)

        # Process results
        for assembly_id, result in zip(correction_tasks.keys(), correction_results):
            if isinstance(result, Exception):
                print(f"Assembly {assembly_id} correction raised an exception: {result}; retaining original labeling.")
            elif result:
                print(f"Assembly {assembly_id} corrected labeling: {result}")
                assembly_pages[assembly_id] = result
            else:
                print(f"Assembly {assembly_id} correction failed; retaining original labeling.")


    # Step 6: Update the DataFrame's "Page" column.
    # For each row in the CSV, if the element ID is found in the consensus results, assign its page.
    for idx, row in df.iterrows():
        elem = row["Element ID"]
        assembly_id = row["Assembly ID"]
        assigned_page = None
        for page, elem_ids in assembly_pages[assembly_id].items():
            if elem in elem_ids:
                if assigned_page is not None:
                    print(
                        f"Warning: Element ID '{elem}' appears on multiple pages for assembly"
                        f" {assembly_id}."
                    )
                assigned_page = page
        if assigned_page is not None:
            df.at[idx, "Page"] = assigned_page
        else:
            print(f"Warning: Asembly {assembly_id} Element ID '{elem}' not found in any page labeling.")

    return df


if __name__ == "__main__":
    csv_path = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    df = load_answer_key(csv_path)
    df = asyncio.run(update_csv_with_pages(df, "data/eval_on/single_images/"))

    df.to_csv("data/fsi_labels/Hadrian Vllm test case - Final Merge - page.csv", index=False)

    # print("Updated CSV with page numbers.")
