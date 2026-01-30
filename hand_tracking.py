import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

def main():
    # Initialize Volume Control
    devices = AudioUtilities.GetSpeakers()
    volume = devices.EndpointVolume
    
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

    print("Starting gesture controller... Press 'q' to quit.")

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

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        thumb_states = [] # List of (is_up, is_down) for each detected hand
        total_fingers = 0

        if result.hand_landmarks:
            for hand_idx, hand_lms in enumerate(result.hand_landmarks):
                # Draw connectors and landmarks
                for start, end in HAND_CONNECTIONS:
                    s_lm, e_lm = hand_lms[start], hand_lms[end]
                    cv2.line(frame, (int(s_lm.x * w), int(s_lm.y * h)), (int(e_lm.x * w), int(e_lm.y * h)), (0, 255, 0), 2)
                for lm in hand_lms:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 255), cv2.FILLED)

                # Finger counting logic
                fingers = []
                
                # Get handedness
                handedness = result.handedness[hand_idx][0].category_name
                
                # Thumb: Tip (4) vs IP (3) or MCP (2) on X-axis
                # Note: 'Right' label in MediaPipe might be mirrored or not depending on camera
                if handedness == 'Right':
                    if hand_lms[4].x < hand_lms[2].x: fingers.append(1)
                    else: fingers.append(0)
                else: # Left
                    if hand_lms[4].x > hand_lms[2].x: fingers.append(1)
                    else: fingers.append(0)

                # Other 4 fingers: Tip (8,12,16,20) vs PIP (6,10,14,18) on Y-axis
                for tid in tip_ids:
                    if hand_lms[tid].y < hand_lms[tid - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers += sum(fingers)

                # Thumb Gesture Detection (Volume/Brightness)
                # Thumb Up: Tip (4) is significantly ABOVE Thumb MCP (2)
                # Thumb Down: Tip (4) is significantly BELOW Thumb MCP (2)
                thumb_up = hand_lms[4].y < hand_lms[2].y - 0.05
                thumb_down = hand_lms[4].y > hand_lms[2].y + 0.05
                
                # Check if other fingers are closed (optional but good for accuracy)
                others_closed = all(hand_lms[tid].y > hand_lms[tid - 2].y for tid in tip_ids)
                
                if others_closed:
                    thumb_states.append((thumb_up, thumb_down))

        # Display Finger Count
        cv2.putText(frame, f"Fingers: {total_fingers}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        num_hands = len(thumb_states)
        
        # Action Logic
        if num_hands == 1:
            up, down = thumb_states[0]
            current_vol = volume.GetMasterVolumeLevelScalar()
            if up:
                new_vol = min(1.0, current_vol + 0.02)
                volume.SetMasterVolumeLevelScalar(new_vol, None)
                cv2.putText(frame, "Volume UP", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif down:
                new_vol = max(0.0, current_vol - 0.02)
                volume.SetMasterVolumeLevelScalar(new_vol, None)
                cv2.putText(frame, "Volume DOWN", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        elif num_hands == 2:
            hand1_up, hand1_down = thumb_states[0]
            hand2_up, hand2_down = thumb_states[1]
            current_bri = sbc.get_brightness()[0]
            
            if hand1_up and hand2_up:
                new_bri = min(100, current_bri + 2)
                sbc.set_brightness(new_bri)
                cv2.putText(frame, "Brightness UP", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif hand1_down and hand2_down:
                new_bri = max(0, current_bri - 2)
                sbc.set_brightness(new_bri)
                cv2.putText(frame, "Brightness DOWN", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Gesture Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
