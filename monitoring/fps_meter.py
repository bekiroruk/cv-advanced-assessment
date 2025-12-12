
from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSMeter:
    """
    Hareketli ortalama FPS hesaplayıcı.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.times: Deque[float] = deque(maxlen=window_size)
        self.last_time = None

    def update(self) -> float:
        """
        Her frame'de çağır.
        Anlık FPS değeri döner.
        """
        now = time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            return 0.0

        dt = now - self.last_time
        self.last_time = now

        self.times.append(dt)
        if len(self.times) < 2:
            return 0.0

        avg_dt = sum(self.times) / len(self.times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0
