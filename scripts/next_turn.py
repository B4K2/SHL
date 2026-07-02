"""Build the next /chat request from the previous request + the API's response.

Usage:
    python scripts/next_turn.py prev_request.json response.json "next user message" > next_request.json

Or pipe the response and pass the previous request inline:
    curl -s $URL/chat -d @req.json | python scripts/next_turn.py req.json - "Yes, go ahead."
"""

import json
import sys


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)

    prev_request_path, response_path, user_message = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(prev_request_path) as f:
        prev_request = json.load(f)

    if response_path == "-":
        response = json.load(sys.stdin)
    else:
        with open(response_path) as f:
            response = json.load(f)

    messages = prev_request.get("messages", [])
    messages.append({"role": "assistant", "content": json.dumps(response)})
    messages.append({"role": "user", "content": user_message})

    json.dump({"messages": messages}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
