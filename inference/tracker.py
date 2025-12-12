# inference/tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .utils import iou


@dataclass
class Track:
    track_id: int
    box: np.ndarray        # [x1,y1,x2,y2]
    score: float
    class_id: int
    age: int = 0           # kaç frame yaşadı
    missed: int = 0        # kaç frame üst üste eşleşmedi


class SimpleIOUTracker:
    """
    Çok basit bir çoklu nesne takipçisi:
      - Her karede detection'larla mevcut track'leri IoU üzerinden eşleştirir.
      - Eşleşmeyen detection'lar için yeni track açar.
      - Eşleşmeyen eski track'leri belirli bir 'missed' eşiğinden sonra siler.
    """

    def __init__(self, iou_thresh: float = 0.5, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.tracks: List[Track] = []
        self.next_id: int = 1

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[Track]:
        """
        boxes: (N,4), scores: (N,), class_ids: (N,)
        return: güncel track listesi
        """
        # Mevcut track'lerin yaşını artır
        for t in self.tracks:
            t.age += 1

        used_det = set()

        # track'leri sırayla detection'lara eşleştir
        for t in self.tracks:
            best_iou = 0.0
            best_j = -1
            for j in range(len(boxes)):
                if j in used_det:
                    continue
                if class_ids[j] != t.class_id:
                    continue
                iou_val = iou(t.box, boxes[j])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_thresh:
                # başarılı eşleşme
                t.box = boxes[best_j]
                t.score = float(scores[best_j])
                t.missed = 0
                used_det.add(best_j)
            else:
                # bu karede eşleşmedi
                t.missed += 1

        # Eşleşmeyen detection'lardan yeni track oluştur
        for j in range(len(boxes)):
            if j in used_det:
                continue
            new_track = Track(
                track_id=self.next_id,
                box=boxes[j],
                score=float(scores[j]),
                class_id=int(class_ids[j]),
                age=1,
                missed=0,
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # Çok uzun süre kaybolan track'leri sil
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return list(self.tracks)
