# %%
from math import ceil
from PIL import Image
import difflib

from typing import Union, List

# Define model-specific parameters
MODEL_PARAMS = {
    "o1": {
        "max_dim": 1024,
        "tile_size": 512,
        "base_tokens": 75,
        "tokens_per_tile": 150,
        "price_per_1m_tokens_in": 15.00,  # $15.00 per 1K tokens
    },
    "o3-mini": {
        "max_dim": 1024,
        "tile_size": 512,
        "base_tokens": 75,
        "tokens_per_tile": 150,
        "price_per_1m_tokens_in": 1.10,  # $1.10 per 1M tokens
    },
    "o3-mini-high": {
        "max_dim": 1024,
        "tile_size": 512,
        "base_tokens": 75,
        "tokens_per_tile": 150,
        "price_per_1m_tokens_in": 1.10,  # $1.10 per 1M tokens
    },
    "gpt-4o": {
        "max_dim": 1024,
        "tile_size": 512,
        "base_tokens": 75,
        "tokens_per_tile": 150,
        "price_per_1m_tokens_in": 2.5,
    },
    "gemini-2.0-flash": {
        "max_dim": 2304,
        "tile_size": 768,
        "tokens_per_tile": 258,
        "price_per_1m_tokens_in": 0.1,  # $0.10 per 1M tokens
    },
    "gemini-2.0-flash-lite": {
        "max_dim": 2304,
        "tile_size": 768,
        "tokens_per_tile": 258,
        "price_per_1m_tokens_in": 0.075,  # $0.075 per 1M tokens
    },
    "gemini-1.5-pro": {
        "max_dim": 2304,
        "tile_size": 768,
        "tokens_per_tile": 258,
        "price_per_1m_tokens_in": 1.4,  # 1.25 frompts <128k, 2.50 prompts >128k
    },
    "gemini-2.0-pro-exp-02-05": {  # pricing is a guess
        "max_dim": 2304,
        "tile_size": 768,
        "tokens_per_tile": 258,
        "price_per_1m_tokens_in": 10,  # guess
    },
}


def find_closest_model(model, model_list):
    return difflib.get_close_matches(model, model_list, n=1, cutoff=0.2)


def calculate_image_tokens_and_cost(image_path, model="o1"):
    if model not in MODEL_PARAMS:
        closest_models = find_closest_model(model, MODEL_PARAMS.keys())
        if closest_models:
            closest_model = closest_models[0]
            print(f"Warning: Model '{model}' not found. Using closest match: '{closest_model}'")
            model = closest_model
        else:
            raise ValueError(f"No similar model found for: {model}")

    params = MODEL_PARAMS[model]

    # Open the image and get its dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Resize if necessary
    if width > params["max_dim"] or height > params["max_dim"]:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = params["max_dim"], int(params["max_dim"] / aspect_ratio)
        else:
            width, height = int(params["max_dim"] * aspect_ratio), params["max_dim"]

    # Calculate the number of tiles
    tiles = ceil(width / params["tile_size"]) * ceil(height / params["tile_size"])

    # Calculate total tokens
    tokens = params["base_tokens"] + params["tokens_per_tile"] * tiles

    # Calculate cost
    cost = tokens * (params["price_per_1m_tokens_in"] / 1e6)

    return tokens, cost


def validate_cost(image_path: Union[str, List[str]], model="o1", cutoff=3):
    if isinstance(image_path, list):
        cost = sum([calculate_image_tokens_and_cost(p, model=model)[1] for p in image_path])
    else:
        _, cost = calculate_image_tokens_and_cost(image_path, model=model)
    if cost > cutoff:
        while True:
            response = input(f"Est cost is {cost}. Continue? (yes/no): ").lower().strip()
            if response == "yes":
                break
            elif response == "no":
                print("Operation cancelled.")
                exit()
            else:
                print("Please enter 'yes' or 'no'.")


# Example usage
if __name__ == "__main__":
    image_path = "data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg4.png"
    model = "o1"  # or "o3-mini", "o3-mini-high"
    tokens, cost = calculate_image_tokens_and_cost(image_path, model)
    print(f"Tokens: {tokens}")
    print(f"Estimated cost: ${cost:.6f}")
