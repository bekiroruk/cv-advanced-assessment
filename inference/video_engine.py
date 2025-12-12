
from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Queue
from typing import Optional

import cv2
import numpy as np

from .detector import Detector
from .tracker import SimpleIOUTracker
from .fusion import tracks_to_dicts


@dataclass
class VideoConfig:
    source: int | str = 0          # 0: webcam, yoksa video path
    detect_every: int = 1          # her N karede bir dedektör
    show: bool = True
    save_path: Optional[str] = None


class VideoEngine:
    """
    Video → Dedektör (periyodik) → Tracker → Overlay → Çıktı boru hattı
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Optional[SimpleIOUTracker] = None,
        config: Optional[VideoConfig] = None,
    ):
        self.detector = detector
        self.tracker = tracker or SimpleIOUTracker()
        self.config = config or VideoConfig()
        self.stop_flag = False

    def run_single_thread(self):
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            raise RuntimeError(f"Video kaynağı açılamadı: {self.config.source}")

        writer = None
        if self.config.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.config.save_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_tracks = []

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Her N karede bir dedektör çalıştır
            if frame_idx % self.config.detect_every == 0:
                det = self.detector(frame)
                tracks = self.tracker.update(det.boxes, det.scores, det.class_ids)
                last_tracks = tracks
            else:
                # Dedektör çalışmıyorsa da mevcut track'leri overlay edelim
                tracks = last_tracks

            # Overlay
            vis_frame = self._draw_tracks(frame.copy(), tracks)

            if writer is not None:
                writer.write(vis_frame)

            if self.config.show:
                cv2.imshow("Edge AI Video", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    def _draw_tracks(self, frame: np.ndarray, tracks) -> np.ndarray:
        for t in tracks:
            x1, y1, x2, y2 = t.box.astype(int)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"ID {t.track_id} cls {t.class_id} conf {t.score:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame


# İsteğe bağlı: multi-thread iskeleti
class MultiThreadedVideoEngine(VideoEngine):
    def run(self):
        frame_q: Queue = Queue(maxsize=5)
        stop_flag = {"value": False}

        def capture_loop():
            cap = cv2.VideoCapture(self.config.source)
            if not cap.isOpened():
                print(f"[Capture] Video kaynağı açılamadı: {self.config.source}")
                stop_flag["value"] = True
                return

            while not stop_flag["value"]:
                ret, frame = cap.read()
                if not ret:
                    break
                if not frame_q.full():
                    frame_q.put(frame)
            cap.release()
            stop_flag["value"] = True

        def infer_loop():
            last_tracks = []
            while not stop_flag["value"] or not frame_q.empty():
                if frame_q.empty():
                    continue
                frame = frame_q.get()
                det = self.detector(frame)
                tracks = self.tracker.update(det.boxes, det.scores, det.class_ids)
                last_tracks = tracks
                vis = self._draw_tracks(frame.copy(), tracks)
                if self.config.show:
                    cv2.imshow("Edge AI Video (MT)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_flag["value"] = True
                        break
            cv2.destroyAllWindows()

        t_capture = threading.Thread(target=capture_loop, daemon=True)
        t_infer = threading.Thread(target=infer_loop, daemon=True)

        t_capture.start()
        t_infer.start()

        t_capture.join()
        t_infer.join()
