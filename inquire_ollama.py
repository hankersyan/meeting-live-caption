"""
CLI tool that sends file content to Ollama using project config.json.

Usage:
    python inquire_ollama.py <input_file> [--output OUTPUT_FILE] [--prompt-file PROMPT_FILE]
"""

import argparse
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_ollama_text(text):
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # Remove explicit reasoning blocks if the model emits them.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    if cleaned.lower().startswith("<think>") and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    return cleaned


def adapt_output_format(model, text):
    """Normalize model output formatting with DeepSeek-specific handling."""
    cleaned = clean_ollama_text(text)
    if not cleaned:
        return ""

    # Normalize line endings and trim redundant blank lines.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    normalized_model = (model or "").strip().lower()
    if not normalized_model.startswith("deepseek"):
        return cleaned

    # DeepSeek sometimes wraps the final answer in markdown code fences.
    if cleaned.startswith("```") and cleaned.endswith("```"):
        fenced = cleaned.split("\n")
        if len(fenced) >= 2:
            cleaned = "\n".join(fenced[1:-1]).strip()

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return ""

    bullet_or_number = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
    if any(bullet_or_number.match(line) for line in lines):
        normalized = []
        for line in lines:
            normalized.append("- " + bullet_or_number.sub("", line).strip())
        return "\n".join(normalized)

    # If DeepSeek returns plain paragraphs, keep output readable and list-oriented.
    return "- " + " ".join(lines)


def parse_ollama_text(data):
    """Extract readable output from different Ollama response shapes."""

    def extract(item):
        if not item:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            parts = [extract(x) for x in item]
            return "\n".join([p for p in parts if p])
        if isinstance(item, dict):
            for key in ("response", "text", "output_text", "output", "body"):
                if key in item and item[key]:
                    return extract(item[key])
            if "message" in item:
                return extract(item["message"])
            if "content" in item:
                return extract(item["content"])
            if "choices" in item and isinstance(item["choices"], list):
                return extract(item["choices"])
            if "outputs" in item and isinstance(item["outputs"], list):
                return extract(item["outputs"])
            if "results" in item and isinstance(item["results"], list):
                return extract(item["results"])
        return ""

    candidates = []
    if isinstance(data, str):
        candidates.append(data)
    if isinstance(data, dict):
        for key in (
            "response",
            "text",
            "output_text",
            "output",
            "message",
            "content",
            "results",
            "choices",
            "outputs",
        ):
            if key in data:
                candidates.append(data[key])
        candidates.append(data)

    for item in candidates:
        out = extract(item)
        cleaned = clean_ollama_text(out)
        if cleaned:
            return cleaned

    if isinstance(data, dict):
        for value in data.values():
            out = extract(value)
            cleaned = clean_ollama_text(out)
            if cleaned:
                return cleaned

    return ""


def request_once(base_url, endpoint, payload, timeout_seconds):
    req = urllib.request.Request(
        url=f"{base_url}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
    return parse_ollama_text(json.loads(body))


def build_generate_payload(model, prompt_text, disable_think=False):
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {"num_predict": 500},
    }
    if disable_think:
        payload["think"] = False
    return payload


def build_chat_payload(model, prompt_text):
    return {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"num_predict": 500},
        "messages": [
            {
                "role": "system",
                "content": "Answer the user prompt from the provided text. Do not include chain-of-thought.",
            },
            {"role": "user", "content": prompt_text},
        ],
    }


def request_with_fallbacks(base_url, model, prompt_text, timeout_seconds):
    normalized_model = model.lower().strip()
    prefers_chat_api = normalized_model.startswith(("qwen", "deepseek"))
    request_plan = []

    if prefers_chat_api:
        request_plan.extend(
            [
                ("/api/chat", build_chat_payload(model, prompt_text)),
                ("/api/generate", build_generate_payload(model, prompt_text, disable_think=True)),
            ]
        )
    else:
        request_plan.extend(
            [
                ("/api/generate", build_generate_payload(model, prompt_text, disable_think=False)),
                ("/api/generate", build_generate_payload(model, prompt_text, disable_think=True)),
                ("/api/chat", build_chat_payload(model, prompt_text)),
            ]
        )

    seen = set()
    for endpoint, payload in request_plan:
        payload_key = json.dumps(payload, sort_keys=True)
        request_key = (endpoint, payload_key)
        if request_key in seen:
            continue
        seen.add(request_key)

        result = request_once(base_url, endpoint, payload, timeout_seconds)
        if result:
            return adapt_output_format(model, result)

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Send input file content to Ollama using this project's config.json"
    )
    parser.add_argument("input_file", help="Path to input text file used as prompt content")
    parser.add_argument("--output", "-o", help="Optional output file path")
    parser.add_argument(
        "--prompt-file",
        "-p",
        help="Optional prompt file path. When provided, its content overrides config.json extract_prompt",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    input_path = os.path.abspath(args.input_file)
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    prompt_file_path = None
    if args.prompt_file:
        prompt_file_path = os.path.abspath(args.prompt_file)
        if not os.path.exists(prompt_file_path):
            print(f"Error: prompt file not found: {prompt_file_path}", file=sys.stderr)
            return 1

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error: failed to load config.json: {e}", file=sys.stderr)
        return 1

    base_url = str(config.get("ollama_url", "")).strip().rstrip("/")
    model = str(config.get("ollama_model", "")).strip()
    instruction = str(config.get("extract_prompt", "")).strip()
    timeout_raw = config.get("extract_timeout", 90)

    try:
        timeout_seconds = max(10.0, float(timeout_raw))
    except (TypeError, ValueError):
        timeout_seconds = 90.0

    if not base_url or not model:
        print("Error: config.json must contain non-empty ollama_url and ollama_model.", file=sys.stderr)
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            file_content = f.read().strip()
    except Exception as e:
        print(f"Error: failed to read input file: {e}", file=sys.stderr)
        return 1

    if not file_content:
        print("Error: input file is empty.", file=sys.stderr)
        return 1

    if prompt_file_path:
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()
        except Exception as e:
            print(f"Error: failed to read prompt file: {e}", file=sys.stderr)
            return 1

    if instruction:
        prompt_text = f"{instruction}\n\nInput text:\n{file_content}\n"
    else:
        prompt_text = file_content

    try:
        result = request_with_fallbacks(base_url, model, prompt_text, timeout_seconds)
        if not result:
            print(
                "Error: Ollama returned no displayable text. Try another model or prompt.",
                file=sys.stderr,
            )
            return 1
    except socket.timeout:
        print(
            f"Error: request timed out after {int(timeout_seconds)}s.",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", "")
        if isinstance(reason, socket.timeout):
            print(
                f"Error: request timed out after {int(timeout_seconds)}s.",
                file=sys.stderr,
            )
        else:
            print(f"Error: Ollama request failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: unexpected failure while querying Ollama: {e}", file=sys.stderr)
        return 1

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base_name}.ollama.txt")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
            f.write("\n")
    except Exception as e:
        print(f"Error: failed to write output file: {e}", file=sys.stderr)
        return 1

    print(result)
    print(f"\nSaved output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())