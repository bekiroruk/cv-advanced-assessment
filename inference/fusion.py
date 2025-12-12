
from __future__ import annotations

from typing import List, Dict

import numpy as np

from .utils import iou
from .tracker import Track


def fuse_detections_and_tracks(
    det_boxes: np.ndarray,
    trk_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> List[Dict]:
    """
    Basit füzyon mantığı:
      - Her track box'ı için en yakın detection box'ı bulur.
      - IoU < thresh ise 'drifted=True' işaretler (yeniden başlatma adayı).
      - IoU >= thresh ise tracker'a güvenilir ('source=tracker').
    """
    fused = []

    for i, tb in enumerate(trk_boxes):
        best_iou = 0.0
        best_j = -1
        for j, db in enumerate(det_boxes):
            val = iou(tb, db)
            if val > best_iou:
                best_iou = val
                best_j = j

        fused.append(
            {
                "track_index": i,
                "matched_det_index": best_j,
                "iou": best_iou,
                "drifted": bool(best_j == -1 or best_iou < iou_thresh),
            }
        )

    return fused


def tracks_to_dicts(tracks: List[Track]) -> List[Dict]:
    """
    API ve video overlay için track'leri dict'e çevirir.
    """
    out = []
    for t in tracks:
        out.append(
            {
                "id": int(t.track_id),
                "box": t.box.tolist(),
                "score": float(t.score),
                "class_id": int(t.class_id),
                "age": int(t.age),
                "missed": int(t.missed),
            }
        )
    return out
