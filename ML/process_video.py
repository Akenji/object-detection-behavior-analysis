# process_video.py
import cv2
import os
import sys
import argparse
import shutil
import numpy as np
import mysql.connector
from datetime import datetime
from ultralytics import YOLO

# ========================
# CONFIGURABLE VARIABLES
# ========================
DB_USER = "catchthem"
DB_PASS = "catchthem@321"
DB_NAME = "exam_monitoring"

# Model paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")

POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")
MOBILE_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolo11n.pt")

# Thresholds for events
LEANING_THRESHOLD = 3
PASSING_THRESHOLD = 3
MOBILE_THRESHOLD = 3

# Action strings
LEANING_ACTION = "Leaning"
PASSING_ACTION = "Passing Paper"
ACTION_MOBILE = "Mobile Phone Detected"

# ========================
# DATABASE CONNECTION
# ========================
try:
    db = mysql.connector.connect(
        host="localhost",
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = db.cursor()
    print("[INFO] Database connected successfully")
except Exception as e:
    print(f"[ERROR] Database connection failed: {e}")
    db = None
    cursor = None

# ========================
# HELPER FUNCTIONS
# ========================
def is_leaning(keypoints):
    """Improved leaning detection by comparing head & shoulder centers."""
    if keypoints is None or len(keypoints) < 7:
        return False

    nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder = keypoints[:7]
    if any(pt is None for pt in [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder]):
        return False

    eye_dist = abs(l_eye[0] - r_eye[0])
    shoulder_dist = abs(l_shoulder[0] - r_shoulder[0])
    shoulder_height_diff = abs(l_shoulder[1] - r_shoulder[1])
    head_center_x = (l_eye[0] + r_eye[0]) / 2
    shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2

    if eye_dist > 0.35 * shoulder_dist:
        return False
    if shoulder_height_diff > 40:
        return False

    return abs(head_center_x - shoulder_center_x) > 60

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_passing_paper(wrists):
    """If any pair of wrists from different people is below threshold => passing paper."""
    threshold = 130
    min_self_wrist_dist = 100
    max_vertical_diff = 100

    for i in range(len(wrists)):
        host = wrists[i]
        if calculate_distance(*host) < min_self_wrist_dist:
            continue
        for j in range(i + 1, len(wrists)):
            guest = wrists[j]
            if calculate_distance(*guest) < min_self_wrist_dist:
                continue

            for hw in host:
                for gw in guest:
                    dist = calculate_distance(hw, gw)
                    vert_diff = abs(hw[1] - gw[1])
                    if dist < threshold and vert_diff < max_vertical_diff:
                        return True
    return False

# ========================
# MAIN VIDEO ANALYSIS
# ========================
def analyze_video(video_path, hall_id):
    """Analyzes a video file for malpractice instead of a live camera feed."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at {video_path}")
        return

    # Load models
    print("[INFO] Loading YOLO models...")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        mobile_model = YOLO(MOBILE_MODEL_PATH)
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Processing video at {fps} FPS")

    # State tracking
    lean_frames = 0
    passing_frames = 0
    mobile_frames = 0
    lean_recording = False
    passing_recording = False
    mobile_recording = False
    lean_video = None
    passing_video = None
    mobile_video = None

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"[PROGRESS] Processing frame {frame_count}/{total_frames}")

            ##frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ===== POSE DETECTION =====
            pose_results = pose_model(frame, conf=0.5, verbose=False)
            leaning_detected = False
            passing_detected = False

            if pose_results and len(pose_results) > 0:
                result = pose_results[0]
                keypoints_data = result.keypoints

                if keypoints_data is not None and len(keypoints_data.xy) > 0:
                    all_wrists = []
                    for person_kps in keypoints_data.xy:
                        person_kps_np = person_kps.cpu().numpy()
                        kps_list = []
                        for kp in person_kps_np:
                            x, y = float(kp[0]), float(kp[1])
                            if x > 0 and y > 0:
                                kps_list.append((x, y))
                            else:
                                kps_list.append(None)

                        if is_leaning(kps_list):
                            leaning_detected = True

                        if len(kps_list) >= 10:
                            left_wrist = kps_list[9]
                            right_wrist = kps_list[10]
                            if left_wrist and right_wrist:
                                all_wrists.append((left_wrist, right_wrist))

                    if len(all_wrists) >= 2:
                        if detect_passing_paper(all_wrists):
                            passing_detected = True

            # ===== LEANING LOGIC =====
            if leaning_detected:
                lean_frames += 1
                if lean_frames >= LEANING_THRESHOLD and not lean_recording:
                    print(f"[ALERT] Leaning detected at frame {frame_count}")
                    lean_recording = True
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    ##lean_video = cv2.VideoWriter("output_leaning.mp4", fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
                    lean_video = cv2.VideoWriter("output_leaning.mp4", fourcc, fps, (frame_width, frame_height))

                if lean_recording and lean_video:
                    lean_video.write(frame)
            else:
                if lean_recording and lean_frames >= LEANING_THRESHOLD:
                    lean_video.release()
                    proof_filename = f"leaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    final_path = os.path.join(MEDIA_DIR, proof_filename)
                    shutil.move("output_leaning.mp4", final_path)
                    
                    if cursor and db:
                        now = datetime.now()
                        date_db = now.strftime("%Y-%m-%d")
                        time_db = now.strftime("%H:%M:%S")
                        sql = """
                            INSERT INTO app_malpraticedetection (date, time, malpractice, proof, lecture_hall_id, verified)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        val = (date_db, time_db, LEANING_ACTION, proof_filename, hall_id, False)
                        cursor.execute(sql, val)
                        db.commit()
                        print(f"[DB] Leaning record inserted: {proof_filename}")

                lean_frames = 0
                lean_recording = False
                lean_video = None

            # ===== PASSING PAPER LOGIC =====
            if passing_detected:
                passing_frames += 1
                if passing_frames >= PASSING_THRESHOLD and not passing_recording:
                    print(f"[ALERT] Passing paper detected at frame {frame_count}")
                    passing_recording = True
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    passing_video = cv2.VideoWriter("output_passing.mp4", fourcc, fps, (frame_width, frame_height))

                if passing_recording and passing_video:
                    passing_video.write(frame)
            else:
                if passing_recording and passing_frames >= PASSING_THRESHOLD:
                    passing_video.release()
                    proof_filename = f"passing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    final_path = os.path.join(MEDIA_DIR, proof_filename)
                    shutil.move("output_passing.mp4", final_path)
                    
                    if cursor and db:
                        now = datetime.now()
                        date_db = now.strftime("%Y-%m-%d")
                        time_db = now.strftime("%H:%M:%S")
                        sql = """
                            INSERT INTO app_malpraticedetection (date, time, malpractice, proof, lecture_hall_id, verified)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        val = (date_db, time_db, PASSING_ACTION, proof_filename, hall_id, False)
                        cursor.execute(sql, val)
                        db.commit()
                        print(f"[DB] Passing paper record inserted: {proof_filename}")

                passing_frames = 0
                passing_recording = False
                passing_video = None

            # ===== MOBILE DETECTION =====
            mobile_results = mobile_model(frame, conf=0.6, verbose=False)
            mobile_detected = False

            if mobile_results and len(mobile_results) > 0:
                for box in mobile_results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = mobile_model.names[cls_id]
                    if label.lower() == "cell phone":
                        mobile_detected = True
                        break

            if mobile_detected:
                mobile_frames += 1
                if mobile_frames >= MOBILE_THRESHOLD and not mobile_recording:
                    print(f"[ALERT] Mobile phone detected at frame {frame_count}")
                    mobile_recording = True
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    mobile_video = cv2.VideoWriter("output_mobile.mp4", fourcc, fps, (frame_width, frame_height))

                if mobile_recording and mobile_video:
                    mobile_video.write(frame)
            else:
                if mobile_recording and mobile_frames >= MOBILE_THRESHOLD:
                    mobile_video.release()
                    proof_filename = f"mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    final_path = os.path.join(MEDIA_DIR, proof_filename)
                    shutil.move("output_mobile.mp4", final_path)
                    
                    if cursor and db:
                        now = datetime.now()
                        date_db = now.strftime("%Y-%m-%d")
                        time_db = now.strftime("%H:%M:%S")
                        sql = """
                            INSERT INTO app_malpraticedetection (date, time, malpractice, proof, lecture_hall_id, verified)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        val = (date_db, time_db, ACTION_MOBILE, proof_filename, hall_id, False)
                        cursor.execute(sql, val)
                        db.commit()
                        print(f"[DB] Mobile detection record inserted: {proof_filename}")

                mobile_frames = 0
                mobile_recording = False
                mobile_video = None

    except KeyboardInterrupt:
        print("[INFO] Analysis interrupted by user")
    finally:
        cap.release()
        if lean_video:
            lean_video.release()
        if passing_video:
            passing_video.release()
        if mobile_video:
            mobile_video.release()
        if db:
            db.close()
        print(f"[INFO] Finished analyzing {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a pre-recorded video for malpractice.")
    parser.add_argument("--video", required=True, help="Path to the video file to analyze.")
    parser.add_argument("--hall_id", required=True, type=int, help="Lecture Hall ID for logging.")
    
    args = parser.parse_args()
    
    analyze_video(args.video, args.hall_id)
