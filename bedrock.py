from __future__ import annotations

import boto3
from botocore.exceptions import ClientError
from typing import Generator


class BedrockClient:
    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model"]["id"]
        self.max_tokens: int = config["model"]["max_tokens"]
        self.temperature: float = config["model"].get("temperature", 0.7)
        self.history: list[dict] = []

        session = boto3.Session(
            profile_name=config["aws"]["profile"],
            region_name=config["aws"]["region"],
        )
        self.client = session.client("bedrock-runtime")

    def chat(self, user_message: str, system_prompt: str | None = None) -> Generator[str, None, None]:
        """Send a message and stream back the response token by token."""
        self.history.append({"role": "user", "content": [{"text": user_message}]})

        kwargs: dict = {
            "modelId": self.model_id,
            "messages": self.history,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse_stream(**kwargs)

        full_text = ""
        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    chunk: str = delta["text"]
                    full_text += chunk
                    yield chunk
            elif "messageStop" in event:
                break

        self.history.append({"role": "assistant", "content": [{"text": full_text}]})

    def reset(self) -> None:
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2
