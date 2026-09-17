import cv2
import numpy as np

# Which AprilTag family to look for. Change this to match the tags you printed.
# Options include: DICT_APRILTAG_16h5, DICT_APRILTAG_25h9,
#                   DICT_APRILTAG_36h10, DICT_APRILTAG_36h11
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
if not cap.isOpened():
    print("Error: could not open video stream.")
    exit(1)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            box = tag_corners[0].astype(np.int32)
            cv2.polylines(frame, [box], isClosed=True, color=(0, 0, 255), thickness=2)

            center = tag_corners[0].mean(axis=0)
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {tag_id} ({cx}, {cy})", (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Detected tag {tag_id} at ({cx}, {cy})")

    cv2.imshow("AprilTag Detection (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
