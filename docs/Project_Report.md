# CatchThem: Real-Time Malpractice Detection & Classroom Behavior Analysis

## Executive Summary
CatchThem is a practical, offline-first AI system that helps institutions safeguard examination integrity by automatically detecting suspicious behaviors and prohibited items in classrooms. Built on a robust Django backend and Ultralytics YOLO models, CatchThem combines object detection with pose estimation to identify events such as mobile phone usage, passing notes, turning back, leaning, and more. It delivers real-time alerts, captures video evidence, and provides a friendly dashboard for review and verification—reducing manual workload while improving fairness, transparency, and compliance.

> Insert Logo/Branding Here

---

## Problem & Value Proposition
- Traditional invigilation is resource-intensive and error-prone—especially in large halls or multi-room exams.
- Subtle, tech-enabled cheating often goes unnoticed without augmented oversight.
- Institutions need reliable, scalable, and privacy-preserving systems that work with existing hardware.

CatchThem addresses these challenges with an edge-first design that runs locally on standard machines, integrates easily with existing workflows, and provides actionable evidence for informed decisions.

---

## Project Objectives
- Detect and flag suspicious actions and prohibited objects in real time.
- Capture short, metadata-rich video proofs tied to classrooms and timestamps.
- Provide an admin dashboard to review, verify, and manage alerts.
- Scale across multiple cameras and classrooms with modular scripts.
- Maintain privacy via local processing and optional secure transfers.

---

## System Overview
CatchThem consists of:
- A Django web application for authentication, user management, alert review, and reporting.
- Modular ML scripts (YOLOv8 object detection + YOLOv8-Pose) for behavior and device detection.
- A MySQL/MariaDB database storing logs, users, classrooms, and review status.
- Optional secure client-server transfer (SCP/SSH) when cameras run on separate machines.

---

## Architecture
- Video capture from one or more cameras (front, top, corner)
- Real-time inference via YOLO models (detectors + pose)
- Event confirmation using frame thresholds
- Automated proof recording and storage in `media/`
- Logging to DB and alerting through the Django admin dashboard
- Review & verification with email notifications to assigned teachers

[Insert Architecture Diagram Here]

---

## Use Case Scenarios
- Student uses a mobile device → object detection flags, records snippet, logs to dashboard.
- Passing notes/turning back → pose estimation detects gestures, generates proof, admin reviews.
- Multi-hall monitoring → assign teachers to halls, filter and review cases per hall and building.
- Edge deployment → camera scripts run on client devices; proofs and logs sync to central server.

[Insert Use Case Diagram Here]

---

## Core Features
- Real-time detection: phones, suspicious gestures (leaning, turning, passing paper).
- Pose estimation: skeletal keypoints to infer posture and interactions.
- Automated evidence: short MP4 snippets with date, time, hall metadata.
- Admin dashboard: login, teacher management, lecture hall assignment, alert filters, reviews.
- Notifications: email alerts on verified malpractice to assigned teachers.
- Scalability: multi-camera, modular scripts for different vantage points.
- Offline-first: runs locally; optional secure network transfer.

---

## Machine Learning Approach
- Models: Ultralytics `YOLOv8` for object detection and `YOLOv8-Pose` for behavior analysis.
- Scripts: `ML/mobile_detection.py`, `ML/passing_paper.py`, `ML/turning_back.py`, `ML/leaning.py`, `ML/hand_raise.py`, `ML/front.py`, `ML/top_corner.py`, `ML/top.py`.
- Thresholding: frame-count based confirmation to reduce false positives.
- Evidence pipeline: conditional recording; saves final proofs to `media/` with timestamped filenames.

[Optional: Insert Model/Dataflow Diagram Here]

---

## Data Model Overview
Key entities (from Django models):
- `LectureHall`: building, hall name, `assigned_teacher`.
- `MalpraticeDetection`: date, time, malpractice type, proof filename, verification status, linked `LectureHall`.
- `TeacherProfile`: user, phone, profile picture, mapped `LectureHall`.

[Insert Class Diagram Here]

---

## Admin Dashboard & User Experience
Primary UI pages (templates):
- Index & Login: entry point and authentication.
- Profile & Edit Profile: teacher details and updates.
- Manage Lecture Halls: assign halls to teachers and manage rooms.
- Run Cameras: guidance to launch scripts for capture.
- Malpractice Log: filter by date/time/building/faculty; review and verify events.

[Insert Screenshot: Home]
[Insert Screenshot: Login]
[Insert Screenshot: Profile]
[Insert Screenshot: Manage Lecture Halls]
[Insert Screenshot: Run Cameras]
[Insert Screenshot: Malpractice Log]

---

## Deployment & Setup (High-Level)
1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Configure DB credentials in settings or environment.
4. Run migrations and start the Django server.
5. Launch camera scripts under `ML/` for the desired viewpoints.
6. Optionally configure SCP/SSH for client devices.

[Insert Screenshot: System Running]

---

## Security & Privacy
- Local-first processing: no cloud dependency by default.
- Role-based access: admin vs. teacher views.
- Event verification: human-in-the-loop to confirm malpractice.
- Optional secure transfer: SCP/SSH for remote camera clients.

---

## Performance & Scalability
- Runs on commodity hardware; benefits from GPUs but not mandatory.
- Modular camera scripts allow parallel, multi-room monitoring.
- Threshold-based confirmation reduces false positives; configurable per script.

---

## Roadmap
- Expand behavior catalog (e.g., zone intrusion, crowd counting analytics).
- Add SMS notifications and richer audit trails.
- Enhance reviewer UX with bulk actions and tags.
- Model fine-tuning with institution-specific datasets.

---

## Impact & ROI
- Improved exam integrity with objective, consistent detection.
- Reduced burden on invigilators; focus on oversight and intervention.
- Actionable evidence enables transparent decision-making.
- Scales across halls without proportionally increasing staff.

---

## Appendix
- Requirements: see `requirements.txt`.
- Configuration: `app/settings.py` and environment variables.
- Media outputs: `media/` folder for recorded proofs.
- Static assets & templates: `static/` and `templates/`.
- Reference: Project README with installation and usage details.

[Insert References/Bibliography]
