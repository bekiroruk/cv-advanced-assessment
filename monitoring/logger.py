
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonLogger:
    """
    Basit JSON satır logger'ı.
    Her log çağrısında dosyaya tek satır JSON yazar.
    """

    def __init__(self, log_path: str = "logs/metrics.jsonl") -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **data: Any) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "event": event_type,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
