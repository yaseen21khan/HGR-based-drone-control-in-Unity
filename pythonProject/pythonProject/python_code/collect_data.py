import os
import cv2
import mediapipe as mp
from time import sleep
import datetime

def capture_gesture(class_name, num_gestures=1, duration_per_gesture=3):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    base_folder = "D:/Shared/for Yaseen/dynamic_gestures1"
    os.makedirs(base_folder, exist_ok=True)

    gesture_folder = os.path.join(base_folder, class_name)
    os.makedirs(gesture_folder, exist_ok=True)

    print(f"Collecting {num_gestures} gestures for class '{class_name}'")
    for gesture_num in range(1, num_gestures + 1):
        timestr = datetime.datetime.now().strftime("%H%M%S")
        gesture_subfolder = os.path.join(gesture_folder, f"{gesture_num:03d}_{timestr}")
        os.makedirs(gesture_subfolder, exist_ok=True)

        print(f"\nCollecting Gesture {gesture_num}...")
        sleep(1)

        cap = cv2.VideoCapture("http://192.168.x.yyy:9090/video")
        # cap.set(3, 640)
        # cap.set(4, 480)

        for second in range(1, duration_per_gesture + 1):
            print(f"{duration_per_gesture - second + 1} seconds remaining...")
            sleep(1)

        for frame_num in range(duration_per_gesture * 30):
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame.")
                break

            # Rotate and rescale the frame
            # scaling_factor = 900 / frame.shape[1]
            # rescaled_frame = cv2.resize(frame, (int(600), int(frame.shape[0] * scaling_factor)))
            #rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            resize_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Gesture Capture', resize_frame)
            cv2.moveWindow('Gesture Capture', 0, 0)
            img_filename = os.path.join(gesture_subfolder, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(img_filename, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()
    print(f"Gesture collection for class '{class_name}' completed.")

if __name__ == "__main__":
    classes_to_collect = ["1 Drone selection", "your label2", "your label2" ... etc]
    for class_name in classes_to_collect:
        capture_gesture(class_name)
