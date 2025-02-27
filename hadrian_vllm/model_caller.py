# model_caller.py
import time
import asyncio
import base64
from collections import deque
import json

import litellm
from litellm import completion

from hadrian_vllm.cache import PersistentCache

litellm.vertex_project = "gen-lang-client-0392240747"

# Rate limits for different models (per minute)
RATE_LIMITS = {
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

# Initialize the persistent cache
response_cache = PersistentCache()


def get_base64_image(image_path):
    """Get the base64 encoded image"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def under_ratelimit(model):
    """Ensure we don't exceed rate limits for a model"""
    async with model_locks[model]:
        current_time = time.time()
        # If we've reached the limit of requests per minute
        if len(model_request_timestamps[model]) >= RATE_LIMITS[model]:
            # Calculate how long until a slot opens up (oldest request + 60 seconds)
            oldest_request = model_request_timestamps[model][0]
            time_until_available = oldest_request + 60 - current_time
            if time_until_available > 0:
                await asyncio.sleep(time_until_available)
        # Add this request's timestamp
        model_request_timestamps[model].append(current_time)


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
        "temperature": 0.2,
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
        "temperature": 0.2,
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


async def call_model(prompt_or_messages, image_paths=None, model="gpt-4o", cache=True):
    """
    Call the model with the given prompt/messages and images.

    Args:
        prompt_or_messages: Either a text prompt or a list of messages
        image_paths: List of paths to images (only used if prompt_or_messages is a string)
        model: The model to use
        cache: Whether to use caching

    Returns:
        The model's response
    """
    # Determine if this is a multi-turn conversation
    is_multiturn = isinstance(prompt_or_messages, list)

    # Create a cache key
    cache_key = get_cache_key(prompt_or_messages, image_paths if not is_multiturn else [], model)

    # Check if we have a cached response
    if cache and cache_key in response_cache:
        return response_cache[cache_key]

    # Ensure we respect rate limits
    await under_ratelimit(model)

    try:
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

        # Make the API call
        response = await asyncio.to_thread(completion, **request)

        # Extract the content
        if model.startswith("claude"):
            content = response.choices[0].message.content
        else:  # OpenAI or Gemini
            content = response.choices[0].message.content

        # Cache the response
        if cache:
            response_cache[cache_key] = content

        return content
    except Exception as e:
        print(f"Error calling {model}: {e}")
        raise e  # GRIB
        return ""


def get_openai_messages(prompt_or_messages, image_paths=None):
    """
    Get the messages list in OpenAI format for litellm.

    Args:
                    prompt_or_messages: Either a text prompt or a list of messages
                    image_paths: List of paths to images (only used if prompt_or_messages is a string)

    Returns:
                    List of message objects in OpenAI format
    """
    # Determine if this is a multi-turn conversation
    is_multiturn = isinstance(prompt_or_messages, list)

    if is_multiturn:
        return prepare_openai_multiturn_request(prompt_or_messages, "gpt-4o")["messages"]
    else:
        return prepare_openai_request(prompt_or_messages, image_paths, "gpt-4o")["messages"]
