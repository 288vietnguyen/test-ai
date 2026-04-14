import os
import yaml

DEFAULTS = {
    "aws": {
        "profile": "default",
        "region": "us-east-1",
    },
    "model": {
        "id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "system_prompt": "You are a helpful AI assistant. Be concise and accurate.",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str = "config.yaml") -> dict:
    config = DEFAULTS.copy()
    if os.path.exists(path):
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    return config
