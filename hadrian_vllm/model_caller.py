# model_caller.py
import time
import asyncio
import base64
from collections import deque, OrderedDict
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

import litellm
from litellm import completion

from hadrian_vllm.cache import PersistentCache
from hadrian_vllm.image_cost import calculate_request_tokens_and_cost

litellm.vertex_project = "gen-lang-client-0392240747"
litellm.drop_params = True  # o1 only allows temp=1, hack in case of others

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Rate limits for different models (per minute)
RATE_LIMITS = {
    "o1": 500,
    "o3-mini": 5000,
    "gpt-4o": 10000,
    "gemini-2.0-pro-exp-02-05": 5,
    "gemini-2.0-flash-001": 2000,
}
TOKEN_LIMITS = {
    "o1": 3000000,
    "o3-mini": 5000000,
    "gpt-4o": 450000,
    "gemini-2.0-pro-exp-02-05": 5000,
    "gemini-2.0-flash-001": 2000000,
}


# Initialize a deque to track tokens used per model in the last 60 seconds.
# (timestamp, tokens)
model_token_usage = {model_name: deque() for model_name in RATE_LIMITS}

# Lock for accessing the timestamps safely in async context
model_locks = {model_name: asyncio.Lock() for model_name in RATE_LIMITS}

# Initialize the persistent cache
response_cache = PersistentCache()


class LimitedSizeDict(OrderedDict):
    def __init__(self, size_limit):
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            self.popitem(last=False)


IMG_CACHE_SIZE = 1000
image_base64_cache = LimitedSizeDict(IMG_CACHE_SIZE)


def _load_and_encode(image_path):
    """Synchronously load an image from disk and encode it in base64."""
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return encoded


def preload_images(image_paths, max_workers=16):
    """
    Preload many images into the cache concurrently.

    :param image_paths: List of image file paths to preload.
    :param max_workers: Number of worker threads to use.
    """
    # Deduplicate image paths

    unique_paths = [i for i in set(image_paths[:IMG_CACHE_SIZE]) if i not in image_base64_cache]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all load tasks concurrently.
        futures = {executor.submit(_load_and_encode, path): path for path in unique_paths}
        for future in futures:
            path = futures[future]
            try:
                encoded = future.result()
                image_base64_cache[path] = encoded
            except Exception as e:
                print(f"Error loading {path}: {e}")


def get_base64_image(image_path):
    """Get the base64 encoded image with caching"""
    if image_path in image_base64_cache:
        return image_base64_cache[image_path]

    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        image_base64_cache[image_path] = base64_data
        return base64_data


async def under_ratelimit(model, new_tokens=100):
    """
    Ensure that both the request count and the token usage for the model do not exceed their per-minute limits.
    new_tokens: the estimated tokens of the new request.
    """
    async with model_locks[model]:
        current_time = time.time()
        # Clean up old token entries (older than 60 seconds)
        while model_token_usage[model] and model_token_usage[model][0][0] + 60 < current_time:
            model_token_usage[model].popleft()
        # Sum tokens used in the last 60 seconds
        current_token_sum = sum(entry[1] for entry in model_token_usage[model])
        tokens_limit = TOKEN_LIMITS.get(model, 1e6)
        if current_token_sum + new_tokens > tokens_limit:
            extra_needed = (current_token_sum + new_tokens) - tokens_limit
            cumulative = 0
            wait_time = 0
            # Iterate over the queue and sum tokens until we have enough that must expire
            for timestamp, tokens in model_token_usage[model]:
                cumulative += tokens
                if cumulative >= extra_needed:
                    wait_time = (timestamp + 60) - current_time
                    break
            if wait_time > 0:
                logger.info(f"{model} token limit reached; sleeping for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                current_time = time.time()

        # Then enforce the existing requests-per-minute limit:
        if len(model_token_usage[model]) >= RATE_LIMITS[model]:
            oldest_request = model_token_usage[model][0]
            time_until_available = oldest_request + 60 - current_time
            if time_until_available > 0:
                if time_until_available > 1:
                    logger.info(
                        f"{model} request limit reached; sleeping for"
                        f" {time_until_available:.2f} seconds"
                    )
                await asyncio.sleep(time_until_available)
                current_time = time.time()
        # Record the new tokens usage with the current timestamp
        model_token_usage[model].append((current_time, new_tokens))


def prepare_openai_request(prompt, image_paths, model):
    """Prepare request for OpenAI models"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Add images to the message content
    for image_path in image_paths:
        image_base64 = get_base64_image(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high",
                },
            }
        )

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2 if model != "o1" else 1,
        "max_tokens": 4000,
    }


def prepare_openai_multiturn_request(messages, model):
    """Prepare request for OpenAI models with multi-turn conversation"""
    formatted_messages = []

    for message in messages:
        if message["role"] == "system":
            formatted_messages.append({"role": "system", "content": message["content"]})
        elif message["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": message["content"]})
        elif message["role"] == "user":
            # User messages may include images
            content = [{"type": "text", "text": message["content"]}]

            if "image_path" in message:
                image_base64 = get_base64_image(message["image_path"])
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    }
                )

            formatted_messages.append({"role": "user", "content": content})

    return {
        "model": model,
        "messages": formatted_messages,
        "temperature": 0.2 if model != "o1" else 1,
        "max_tokens": 4000,
    }


def prepare_anthropic_request(prompt, image_paths, model):
    """Prepare request for Anthropic models"""
    content = [{"type": "text", "text": prompt}]

    # Add images to the message content
    for image_path in image_paths:
        image_base64 = get_base64_image(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64,
                },
            }
        )

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 4000,
    }


def prepare_anthropic_multiturn_request(messages, model):
    """Prepare request for Anthropic models with multi-turn conversation"""
    formatted_messages = []

    # Extract system message if present
    system_content = None
    for message in messages:
        if message["role"] == "system":
            system_content = message["content"]
            break

    # Process non-system messages
    for message in messages:
        if message["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": message["content"]})
        elif message["role"] == "user":
            # User messages may include images
            content = [{"type": "text", "text": message["content"]}]

            if "image_path" in message:
                image_base64 = get_base64_image(message["image_path"])
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    }
                )

            formatted_messages.append({"role": "user", "content": content})

    request = {
        "model": model,
        "messages": formatted_messages,
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    if system_content:
        request["system"] = system_content

    return request


def prepare_gemini_request(prompt, image_paths, model):
    """Prepare request for Gemini models"""
    content = [{"type": "text", "text": prompt}]

    # Add images to the message content
    for image_path in image_paths:
        image_base64 = get_base64_image(image_path)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        )

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 4000,
    }


def prepare_gemini_multiturn_request(messages, model):
    """Prepare request for Gemini models with multi-turn conversation"""
    formatted_messages = []

    for message in messages:
        if message["role"] == "system":
            # Gemini doesn't support system messages directly; prepend to first user message
            continue
        elif message["role"] == "assistant":
            # formatted_messages.append({"role": "model", "content": message["content"]})
            formatted_messages.append(
                {"role": "assistant", "content": message["content"]}
            )  # for litellm
        elif message["role"] == "user":
            # User messages may include images
            content = [{"type": "text", "text": message["content"]}]

            if "image_path" in message:
                image_base64 = get_base64_image(message["image_path"])
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    }
                )

            formatted_messages.append({"role": "user", "content": content})

    # If there's a system message, prepend it to the first user message
    system_content = None
    for message in messages:
        if message["role"] == "system":
            system_content = message["content"]
            break

    if system_content and formatted_messages:
        for i, msg in enumerate(formatted_messages):
            if msg["role"] == "user":
                if (
                    isinstance(msg["content"], list)
                    and msg["content"]
                    and isinstance(msg["content"][0], dict)
                    and msg["content"][0]["type"] == "text"
                ):
                    msg["content"][0]["text"] = f"{system_content}\n\n{msg['content'][0]['text']}"
                    break

    return {
        "model": model,
        "messages": formatted_messages,
        "temperature": 0.2,
        "max_tokens": 4000,
    }


def get_cache_key(prompt_or_messages, image_paths, model):
    """Generate a cache key based on the request parameters"""
    if isinstance(prompt_or_messages, str):
        # Single prompt with image paths
        return f"{model}:{prompt_or_messages}:{','.join(image_paths)}"
    else:
        # Multi-turn messages
        # Extract image paths from messages
        img_paths = []
        for msg in prompt_or_messages:
            if "image_path" in msg:
                img_paths.append(msg["image_path"])

        # Serialize messages without image_path for the cache key
        serialized_msgs = json.dumps(
            [{k: v for k, v in msg.items() if k != "image_path"} for msg in prompt_or_messages]
        )

        return f"{model}:{serialized_msgs}:{','.join(img_paths)}"


async def call_model(
    prompt_or_messages,
    image_paths=None,
    model="gpt-4o",
    cache=True,
    max_retries=3,
    retry_base_delay=2,
    return_error_message=True,
):
    """
    Call the model with the given prompt/messages and images.

    Args:
        prompt_or_messages: Either a text prompt or a list of messages
        image_paths: List of paths to images (only used if prompt_or_messages is a string)
        model: The model to use
        cache: Whether to use caching
        max_retries: Maximum number of retry attempts
        retry_base_delay: Base delay between retries (exponential backoff)
        return_error_message: If True, return error message instead of raising exception

    Returns:
        The model's response or error message if return_error_message=True
    """
    # Determine if this is a multi-turn conversation
    is_multiturn = isinstance(prompt_or_messages, list)

    # Create a cache key
    cache_key = get_cache_key(prompt_or_messages, image_paths if not is_multiturn else [], model)

    # Check if we have a cached response
    if cache and cache_key in response_cache:
        cached_response = response_cache[cache_key]
        # Make sure we're not caching errors
        if (
            cached_response
            and not cached_response.startswith("Error:")
            and cached_response != "None"
        ):
            return cached_response

    total_tokens, total_cost = calculate_request_tokens_and_cost(
        prompt_or_messages, image_paths or [], model
    )
    logger.info(f"Estimated tokens for request: {total_tokens}, estimated cost: ${total_cost:.6f}")

    # Prepare the request based on the model type and conversation style
    if is_multiturn:
        if model.startswith("claude"):
            request = prepare_anthropic_multiturn_request(prompt_or_messages, model)
        elif model.startswith("gemini"):
            request = prepare_gemini_multiturn_request(prompt_or_messages, model)
        else:  # OpenAI models
            request = prepare_openai_multiturn_request(prompt_or_messages, model)
    else:
        if model.startswith("claude"):
            request = prepare_anthropic_request(prompt_or_messages, image_paths, model)
        elif model.startswith("gemini"):
            request = prepare_gemini_request(prompt_or_messages, image_paths, model)
        else:  # OpenAI models
            request = prepare_openai_request(prompt_or_messages, image_paths, model)

    # Attempt with retries
    for attempt in range(max_retries):
        try:
            await under_ratelimit(model, new_tokens=total_tokens)
            response = await asyncio.to_thread(
                lambda: completion(**request, timeout=60)
                # completion, **request, timeout=60
            )  # timeout was 10min by default
        except Exception as e:
            error_msg = f"Error calling {model} (attempt {attempt+1}/{max_retries}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            logger.info(e.__traceback__)

            if attempt < max_retries - 1:
                # Calculate backoff delay with jitter to avoid thundering herd
                delay = retry_base_delay * (2**attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                # All attempts failed
                error_response = (
                    f"Error: Failed to get response from {model} after {max_retries} attempts. Last"
                    f" error: {str(e)}"
                )
                logger.error(error_response)

                if return_error_message:
                    # Cache the error to avoid repeated failures
                    if cache:
                        response_cache[cache_key] = error_response
                    return error_response
                else:
                    raise RuntimeError(error_response) from e
        content = response.choices[0].message.content
        if cache and content:
            response_cache[cache_key] = content
        return content
