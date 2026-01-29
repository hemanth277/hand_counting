import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def main():
    # 1. Initialize detector
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 2. Initialize webcam
    cap = cv2.VideoCapture(0)

    p_time = 0
    c_time = 0

    print("Starting finger counter... Press 'q' to quit.")

    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (9, 10), (10, 11), (11, 12),         # Middle
        (13, 14), (14, 15), (15, 16),        # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (5, 9), (9, 13), (13, 17)            # Palm
    ]
    
    # TIP indices: Index, Middle, Ring, Pinky
    tip_ids = [8, 12, 16, 20]

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mirror the frame for a more natural interaction
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                # Get handedness for this hand
                # result.handedness is a list of lists of Category objects
                handedness = result.handedness[i][0].category_name # 'Left' or 'Right'
                
                h, w, _ = frame.shape
                
                # Draw connections
                for start, end in HAND_CONNECTIONS:
                    s_lm = hand_lms[start]
                    e_lm = hand_lms[end]
                    cv2.line(frame, 
                             (int(s_lm.x * w), int(s_lm.y * h)), 
                             (int(e_lm.x * w), int(e_lm.y * h)), 
                             (0, 255, 0), 2)
                
                # Draw landmarks
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # --- Finger Counting Logic ---
                fingers = []

                # 1. Thumb logic
                # For a mirrored image:
                # If Right hand: Thumb tip (4) is to the LEFT of joint (2) when open.
                # If Left hand: Thumb tip (4) is to the RIGHT of joint (2) when open.
                # Note: MediaPipe handedness is flipped for mirrored image sometimes, 
                # but 'Left'/'Right' usually refers to the actual hand.
                if handedness == "Right":
                    if hand_lms[4].x < hand_lms[2].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else: # Left hand
                    if hand_lms[4].x > hand_lms[2].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # 2. Other 4 fingers
                for tid in tip_ids:
                    # Check if tip y is above PIP joint y (lower y in screen coords)
                    if hand_lms[tid].y < hand_lms[tid - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                hand_fingers_count = fingers.count(1)
                total_fingers += hand_fingers_count
                
                # Display count per hand (optional)
                cx, cy = int(hand_lms[0].x * w), int(hand_lms[0].y * h)
                cv2.putText(frame, f"{hand_fingers_count}", (cx, cy - 20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Draw total count (Background removed)
        cv2.putText(frame, str(total_fingers), (45, 410), cv2.FONT_HERSHEY_PLAIN, 
                    10, (255, 0, 0), 20)

        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Show Output
        cv2.imshow("Finger Counter", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
