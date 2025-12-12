
import numpy as np

from inference.tracker import SimpleIOUTracker


def test_tracker_maintains_id_for_same_object():
    tracker = SimpleIOUTracker()

    # İlk frame - tek bbox
    boxes1 = np.array([[100, 100, 200, 200]], dtype=float)
    scores1 = np.array([0.9], dtype=float)
    classes1 = np.array([0], dtype=int)

    tracks1 = tracker.update(boxes1, scores1, classes1)
    assert len(tracks1) == 1
    first_id = tracks1[0].track_id

    # İkinci frame - biraz kaymış ama aynı obje
    boxes2 = np.array([[105, 105, 205, 205]], dtype=float)
    scores2 = np.array([0.88], dtype=float)
    classes2 = np.array([0], dtype=int)

    tracks2 = tracker.update(boxes2, scores2, classes2)
    assert len(tracks2) == 1
    second_id = tracks2[0].track_id

    assert first_id == second_id
