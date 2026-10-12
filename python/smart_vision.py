#!/usr/bin/env python3
import sys
import os
import base64
import json
import mimetypes
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_CONFIG_PATH = PROJECT_ROOT / ".local" / "openai.conf"

DEFAULT_MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 160
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

def load_settings_from_config():
    settings = {
        "api_key": os.environ.get("OPENAI_API_KEY", "").strip(),
        "model": DEFAULT_MODEL,
        "auth_mode": "chatgpt",
        "chatgpt_model": ""
    }

    config_path = str(PRIVATE_CONFIG_PATH)
    script_dir = os.path.dirname(config_path)
    api_key_file = ""

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        key = k.strip()
                        val = v.strip()
                        if key == "openai_api_key" and not settings["api_key"]:
                            settings["api_key"] = val
                        elif key == "openai_model" and val:
                            settings["model"] = val
                        elif key == "openai_api_key_file":
                            api_key_file = val
                        elif key == "ai_auth_mode" and val:
                            settings["auth_mode"] = val
                        elif key == "chatgpt_model":
                            settings["chatgpt_model"] = val
        except Exception:
            pass
    if not settings["api_key"] and api_key_file:
        if not os.path.isabs(api_key_file):
            api_key_file = os.path.join(script_dir, api_key_file)
        try:
            with open(api_key_file, "r") as f:
                settings["api_key"] = f.read().strip()
        except OSError:
            pass
    return settings

def stream_description(response):
    """Accept a description only after the stream's successful terminal event."""
    event_data = []
    text_parts = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            event_data.append(line[5:].lstrip())
        elif not line and event_data:
            data = "\n".join(event_data)
            event_data = []
            if data == "[DONE]":
                break
            event = json.loads(data)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                text_parts.append(event.get("delta", ""))
            elif event_type == "response.refusal.delta":
                return "Unable to analyze scene: ChatGPT could not describe this image."
            elif event_type == "response.completed":
                result = event.get("response", {})
                if result.get("status") != "completed" or result.get("error"):
                    return "Unable to analyze scene: ChatGPT did not complete the description."
                text = " ".join("".join(text_parts).replace("*", "").replace("#", "").split())
                return text or "Unable to analyze scene: ChatGPT returned no description."
            elif event_type in ("response.failed", "error"):
                error = event.get("response", {}).get("error") or event.get("error") or event
                code = error.get("code", "")
                if code in ("subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable"):
                    return "Unable to analyze scene: ChatGPT plan or app usage limit reached. Review ChatGPT Settings, Usage."
                if code in ("subscription_sharing_invalid_user", "chatpass_v2_invalid_authorization_context"):
                    return "Unable to analyze scene: Your ChatGPT connection needs renewing. Press L to sign in again."
                return "Unable to analyze scene: ChatGPT could not complete the request. Please try again."
            elif event_type == "response.incomplete":
                return "Unable to analyze scene: ChatGPT did not finish the description. Please try again."
    return "Unable to analyze scene: The ChatGPT connection ended before the description was complete. Please try again."


def analyze_image_with_openai(image_path, api_key, model, use_chatgpt_plan=False):
    try:
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    except OSError:
        return "Unable to analyze scene: The image file could not be read."

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    prompt = (
        "You are an assistive vision assistant for someone who is visually impaired. "
        "Look closely at what the person is holding in their hand directly in front of the camera. "
        "Read visible text and identify the brand, flavor, color, and object type when clear. "
        "If text is blurry or an object is unclear, say so instead of guessing. "
        "State what they are holding in 1-2 natural, spoken sentences, under 60 words. "
        "Do not use markdown formatting, asterisks, bullet points, or bold text."
    )

    payload = {
        "model": model,
        "instructions": prompt,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{img_b64}",
                "detail": "high"
            }]
        }],
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False
    }
    if use_chatgpt_plan:
        # ChatGPT plan requests require streaming and reject max_output_tokens.
        payload.pop("max_output_tokens")
        payload["stream"] = True
    req = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if use_chatgpt_plan:
                return stream_description(response)
            res_data = json.loads(response.read().decode("utf-8"))

        if res_data.get("status") == "incomplete":
            return "Unable to analyze scene: OpenAI did not finish the description. Please try again."
        if res_data.get("status") == "failed" or res_data.get("error"):
            return "Unable to analyze scene: OpenAI could not complete the request. Please try again."

        text_parts = []
        for item in res_data.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    return "Unable to analyze scene: OpenAI could not describe this image."
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
        text = " ".join(text_parts).replace("*", "").replace("#", "")
        text = " ".join(text.split())
        return text or "Unable to analyze scene: OpenAI returned no description."
    except urllib.error.HTTPError as e:
        e.close()
        if use_chatgpt_plan:
            messages = {
                401: "Your ChatGPT connection needs renewing. Press L to sign in again.",
                403: "ChatGPT plan access was denied. Check your account permission and selected model.",
                429: "ChatGPT plan or app usage limit reached. Review ChatGPT Settings, Usage."
            }
            return "Unable to analyze scene: " + messages.get(e.code, f"ChatGPT returned HTTP {e.code}. Please try again.")
        messages = {
            401: "The OpenAI API key is invalid. Check OPENAI_API_KEY or openai_api_key in .local/openai.conf.",
            403: "The OpenAI project does not have permission to use this model.",
            404: f"The OpenAI model {model} is unavailable. Check model access and openai_model in .local/openai.conf.",
            429: "OpenAI usage or rate limit reached. Check API billing or try again later."
        }
        return "Unable to analyze scene: " + messages.get(e.code, f"OpenAI returned HTTP {e.code}. Please try again.")
    except OSError:
        return "Unable to analyze scene: Could not reach OpenAI. Check your connection and try again."
    except (ValueError, TypeError, AttributeError):
        return "Unable to analyze scene: OpenAI returned an invalid response. Please try again."

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 smart_vision.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isabs(image_path):
        image_path = str(PROJECT_ROOT / image_path)

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    settings = load_settings_from_config()
    if settings.get("auth_mode") == "chatgpt":
        from chatgpt_auth import AuthError, access_token, choose_model
        try:
            token = access_token()
            model = choose_model(token, settings.get("chatgpt_model", ""))
            print(analyze_image_with_openai(image_path, token, model, use_chatgpt_plan=True))
        except AuthError as error:
            print(str(error))
        return
    if settings.get("auth_mode", "api_key") != "api_key":
        print("Unknown AI sign-in mode. Set ai_auth_mode to chatgpt or api_key in .local/openai.conf.")
        return
    api_key = settings.get("api_key")
    model = settings.get("model", DEFAULT_MODEL)

    if not api_key:
        print("To activate Smart Vision AI, set OPENAI_API_KEY or add your OpenAI API key to .local/openai.conf under openai_api_key.")
        sys.exit(0)

    description = analyze_image_with_openai(image_path, api_key, model)
    print(description)

if __name__ == "__main__":
    main()
