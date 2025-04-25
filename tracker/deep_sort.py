# tracker/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=5)

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)