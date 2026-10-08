"""A simple example of sending a question to an AI model through OpenRouter."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"


def get_api_key():
    # Check the environment variable first.
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    # For the workshop, also look for the key in a .env file next to this script.
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def main():
    api_key = get_api_key()
    if not api_key:
        print("Set OPENROUTER_API_KEY in your environment or in a .env file.")
        return

    question = input("Ask the AI a question: ")

    # The JSON request contains the model and the messages to send.
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer clearly and concisely."},
            {"role": "user", "content": question},
        ],
    }

    request = Request(
        API_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    # Send the request, then read the JSON returned by the API.
    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    # The model's answer is in the first item in choices.
    answer = result["choices"][0]["message"]["content"]
    print("\nAI response:")
    print(answer)


if __name__ == "__main__":
    main()
