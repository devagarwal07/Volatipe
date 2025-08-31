from pathlib import Path
from typing import Any, Dict
import yaml
import os

class ConfigLoader:
    def __init__(self, base_dir: str = "config"):
        self.base_dir = Path(base_dir)

    def load(self, filename: str) -> Dict[str, Any]:
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return self._resolve_env(data)

    def _resolve_env(self, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: self._resolve_env(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_env(v) for v in node]
        if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
            inner = node[2:-1]
            if ':-' in inner:
                var, default = inner.split(':-', 1)
                return os.getenv(var, default)
            return os.getenv(inner, node)
        return node

config_loader = ConfigLoader()
