import os
from ultralytics import YOLO
import cv2
import time
import numpy as np
from utils.visualizer import draw_box, draw_fps, draw_status_info

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Load YOLO model
model = YOLO("yolov8s.pt")

# Load video
video_path = "input/test_video1.mp4"
cap = cv2.VideoCapture(video_path)

# Output video writer setup
output_path = "output/processed_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps_out = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps_out, (frame_width, frame_height))

# Object tracking history
prev_tracks = {}
current_tracks = {}
track_history = {}
missing_tracks = {}
new_tracks = {}
object_features = {}

# Enhanced re-identification parameters
missing_threshold = 15
reappear_window = 1000
reidentification_threshold = 0.65
fps_list = []

# Keep track of "true" object IDs
next_true_id = 1
track_to_true_id = {}
class_instance_count = {}
true_id_colors = {}

frame_count = 0

def extract_features(frame, box):
    x1, y1, x2, y2 = box
    try:
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        resized = cv2.resize(roi, (64, 64))
        
        hist_color = cv2.calcHist([resized], [0, 1, 2], None, [16, 16, 16], 
                                [0, 256, 0, 256, 0, 256])
        hist_color = cv2.normalize(hist_color, hist_color).flatten()
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        hist_edges = cv2.calcHist([edges], [0], None, [32], [0, 256])
        hist_edges = cv2.normalize(hist_edges, hist_edges).flatten()
        
        features = np.concatenate((hist_color, hist_edges))
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def compare_features(features1, features2):
    if features1 is None or features2 is None:
        return 0
    try:
        correlation = cv2.compareHist(features1[:256], features2[:256], cv2.HISTCMP_CORREL)
        
        if len(features1) > 256 and len(features2) > 256:
            edge_similarity = cv2.compareHist(features1[256:], features2[256:], cv2.HISTCMP_CORREL)
            return 0.7 * correlation + 0.3 * edge_similarity
        return correlation
    except Exception as e:
        print(f"Feature comparison error: {e}")
        return 0

def get_true_id(track_id, label, features, frame_count):
    global next_true_id
    
    if track_id in track_to_true_id:
        return track_to_true_id[track_id]
    
    best_match_id = None
    best_match_score = reidentification_threshold
    
    for old_track_id, info in list(missing_tracks.items()):
        if info['label'] != label:
            continue
        if frame_count - info['frame'] > reappear_window:
            continue
        if 'features' in info and info['features'] is not None and features is not None:
            similarity = compare_features(features, info['features'])
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = track_to_true_id.get(old_track_id)
                if similarity > 0.85:
                    break
    
    if best_match_id is not None:
        print(f"Re-identified {label} as true ID #{best_match_id} (similarity: {best_match_score:.2f})")
        track_to_true_id[track_id] = best_match_id
        return best_match_id
        
    true_id = next_true_id
    next_true_id += 1
    true_id_colors[true_id] = (
        int((true_id * 57) % 255),
        int((true_id * 121) % 255),
        int((true_id * 233) % 255)
    )
    track_to_true_id[track_id] = true_id
    if label not in class_instance_count:
        class_instance_count[label] = 1
    else:
        class_instance_count[label] += 1
    return true_id

print(f"Processing video: {video_path}")
print(f"Stats: {frame_height}x{frame_width} at {fps_out} FPS")

try:
    model.tracker = "botsort.yaml"
    model.conf = 0.5  
    model.iou = 0.4   
except Exception as e:
    print(f"Warning: Could not set tracker parameters: {e}")
    print("Continuing with default tracker...")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    results = model.track(frame, persist=True, verbose=False)
    current_tracks = {}
    
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = box
            label = results[0].names[cls_id]
            width = x2 - x1
            height = y2 - y1
            features = extract_features(frame, (x1, y1, x2, y2))
            true_id = get_true_id(track_id, label, features, frame_count)
            
            current_tracks[track_id] = {
                'box': (x1, y1, width, height),
                'label': label,
                'last_seen': frame_count,
                'features': features,
                'confidence': conf,
                'true_id': true_id
            }
            
            if true_id not in track_history:
                track_history[true_id] = []
                
                if frame_count > 5:  
                    if track_id not in prev_tracks and true_id not in [info.get('true_id') for info in prev_tracks.values()]:
                        new_tracks[true_id] = {
                            'label': label,
                            'frame': frame_count,
                            'time': time.time(),
                            'features': features
                        }
                        print(f"New {label} detected: True ID #{true_id}")
            
            if len(track_history[true_id]) > 30:
                track_history[true_id].pop(0)
            track_history[true_id].append((x1, y1, x2, y2))
            if true_id in object_features and object_features[true_id] is not None and features is not None:
                object_features[true_id] = 0.7 * object_features[true_id] + 0.3 * features
            else:
                object_features[true_id] = features
            try:
                draw_box(
                    frame, 
                    true_id,
                    x1, y1, 
                    width, height, 
                    label=label, 
                    color=true_id_colors.get(true_id, (0, 255, 0))
                )
            except Exception as e:
                print(f"Error drawing box: {e}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} #{true_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    for track_id, info in prev_tracks.items():
        if track_id not in current_tracks:
            true_id = info.get('true_id')
            if true_id and true_id not in [t.get('true_id') for t in current_tracks.values()]:
                if true_id not in missing_tracks:
                    last_seen = info['last_seen']
                    if frame_count - last_seen >= missing_threshold:
                        missing_tracks[track_id] = {
                            'label': info['label'],
                            'frame': frame_count,
                            'time': time.time(),
                            'features': info.get('features'),
                            'true_id': true_id
                        }
                        print(f"{info['label']} missing: True ID #{true_id}")
                    else:
                        current_tracks[track_id] = info
    active_ids = set(info.get('true_id') for info in current_tracks.values())
    missing_ids = set(info.get('true_id') for info in missing_tracks.values())
    new_object_ids = set(new_tracks.keys())
    try:
        draw_status_info(frame, len(active_ids), len(new_object_ids), len(missing_ids))
    except Exception as e:
        print(f"Error drawing status info: {e}")
    prev_tracks = current_tracks.copy()
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)
    try:
        draw_fps(frame, fps)
    except Exception as e:
        print(f"Error drawing FPS: {e}")
    try:
        resized = cv2.resize(frame, (960, 540))
        cv2.imshow("Object Monitor", resized)
    except Exception as e:
        print(f"Error displaying frame: {e}")
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

class_totals = {}
for true_id in track_history:
    label = None
    for info in current_tracks.values():
        if info.get('true_id') == true_id:
            label = info['label']
            break
    if not label:
        for info in missing_tracks.values():
            if info.get('true_id') == true_id:
                label = info['label']
                break
    if label:
        class_totals[label] = class_totals.get(label, 0) + 1

avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"\nAverage FPS: {avg_fps:.2f}")
print(f"Total Unique Objects: {len(track_history)}")
print(f"Objects by class: {class_totals}")
print(f"New Objects Detected: {len(set(new_tracks.keys()))}")
print(f"Objects Gone Missing: {len(set(info.get('true_id') for info in missing_tracks.values()))}")
print(f"Output video saved at: {output_path}")