import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import json
import pyautogui
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import csv
import copy
import random
import math
import itertools
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

base_options_hand = python.BaseOptions(model_asset_path='hand_landmarker.task')
options_hand = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    num_hands=1, min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7, min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector_hand = vision.HandLandmarker.create_from_options(options_hand)
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils


# def draw_landmarks_on_image(rgb_image, detection_result, prediction, confidence, brect):
#     hand_landmarks_list = detection_result.hand_landmarks
#     annotated_image = np.copy(rgb_image)
#     for idx in range(len(hand_landmarks_list)):
#         mp_drawing.draw_landmarks(annotated_image, hand_landmarks_list[idx], mp_hands.HAND_CONNECTIONS)
#         if len(brect) > 0:      
#             cv.putText(annotated_image, f"{prediction}({confidence:.3f})",(brect[0],brect[1]-5),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),1)
#     return annotated_image

def draw_landmarks_on_image(rgb_image, detection_result,prediction,confidence,brect):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS)
    if len(brect) > 0:      
      cv.putText(annotated_image, f"{prediction}({confidence:.3f})",(brect[0],brect[1]-5),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),1)

  return annotated_image

def get_landmarks(image):
    return image.hand_landmarks #list of lists

# def normalize(hand_landmarks):
#     temp_landmark_list = copy.deepcopy(hand_landmarks)
#     base_x, base_y = 0, 0
#     points = []
#     for i, landmark in enumerate(temp_landmark_list):
#         px = landmark.x if hasattr(landmark, 'x') else landmark[0]
#         py = landmark.y if hasattr(landmark, 'y') else landmark[1]
#         points.append([px, py])
#         if i == 0: base_x, base_y = px, py
#     for point in points:
#         point[0] = point[0] - base_x
#         point[1] = point[1] - base_y
#     max_value = max(list(map(abs, itertools.chain.from_iterable(points))))
#     if max_value == 0: max_value = 1
#     for point in points:
#         point[0] = point[0] / max_value
#         point[1] = point[1] / max_value
#     return list(itertools.chain.from_iterable(points))

def normalize(hand_landmarks):
    temp_landmark_list = copy.deepcopy(hand_landmarks)

    base_x, base_y = 0, 0
    points = []

    for i,landmark in enumerate(temp_landmark_list):
        px = landmark.x if hasattr(landmark, 'x') else landmark[0]
        py = landmark.y if hasattr(landmark, 'y') else landmark[1]

        points.append([px, py])

        if i == 0:
            base_x, base_y = px, py

    for point in points:
        point[0] = point[0] - base_x
        point[1] = point[1] - base_y

    max_value = max(list(map(abs, itertools.chain.from_iterable(points))))

    if max_value == 0: 
        max_value = 1

    def normalize_(n):
        return n / max_value

    for point in points:
        point[0] = normalize_(point[0])
        point[1] = normalize_(point[1])

    flattened_list = list(itertools.chain.from_iterable(points))

    return flattened_list

def get_canonical_landmarks(landmarks, handedness):
    canonical_points = []
    for landmark in landmarks:
        x = landmark.x
        if handedness: x = 1.0 - x # Flip if left hand
        canonical_points.append([x, landmark.y])
    return canonical_points

def augment_landmarks_list(landmark_list):
    augmented_rows = [landmark_list]
    for _ in range(4): 
        new_landmarks = []
        theta = math.radians(random.uniform(-8, 8))
        c, s = math.cos(theta), math.sin(theta)
        scale = random.uniform(0.9, 1.1)
        for i in range(0, len(landmark_list), 2):
            x, y = landmark_list[i], landmark_list[i+1]
            x_new = ((x * c) - (y * s)) * scale
            y_new = ((x * s) + (y * c)) * scale
            new_landmarks.extend([x_new, y_new])
        augmented_rows.append(new_landmarks)
    return augmented_rows

def train_custom_model(user):
    model_save_path = f"model/{user}_model.pkl"
    csv_path = f"model/{user}_data.csv"
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        return None

    X = df.iloc[:, 1:].values  
    y = df.iloc[:, 0].values
    unique_classes = np.unique(y)
    
    if len(unique_classes) < 2:
        print(f"Need at least 2 gestures to train. Current: {unique_classes}")
        return None 

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    except ValueError:
        X_train, y_train = X, y

    model = SVC(kernel='linear', probability=True) 
    model.fit(X_train, y_train)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model retrained for {user}")
    return model

current_user = None
current_hand_model = None
current_label = ""
recording_mode = False          
frames_recorded = 0 
is_counting_down = False
countdown_start_time = 0

mouse = MouseController()
keyboard = KeyboardController()

screen_w, screen_h = pyautogui.size()
prev_screen_x, prev_screen_y = 0,0
last_click_time = 0
click_cooldown = 0.5
freeze_cursor = False

cap = cv.VideoCapture(0)
cap.set(3, 1024)
cap.set(4, 1024)


if not os.path.exists("control.json"):
    with open("control.json", "w") as f:
        json.dump({"record": False, "retrain": False}, f)

print("System Ready. Waiting for commands from dashboard...")

while True:
    key = cv.waitKey(5) & 0xFF
    if key == ord('q') or key == 27: break

    try:
        with open("control.json", "r") as f:
            command = json.load(f)
            
            if command.get("retrain") == True:
                user_to_train = command.get("user_id")
                current_user = user_to_train
                current_hand_model = train_custom_model(current_user)
                with open("control.json", "w") as fw:
                    json.dump({"record": False, "retrain": False}, fw)

            elif command.get("record") == True:
                current_user = command.get("user_id")
                current_label = command.get("gesture_name")
                
                csv_path = f"model/{current_user}_data.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, header=None)
                    df_filtered = df[df[0] != current_label]
                    df_filtered.to_csv(csv_path, index=False, header=False)
                
                # Start countdown
                is_counting_down = True
                countdown_start_time = time.time()
                frames_recorded = 0
                
                with open("control.json", "w") as fw:
                    json.dump({"record": False, "retrain": False}, fw)
                    
    except Exception:
        pass
    
    success, img = cap.read()
    if not success: continue
        
    debug_image = copy.deepcopy(img)
    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    hand_result = detector_hand.detect_for_video(mp_image, int(time.time() * 1000))
    
    if is_counting_down:
        elapsed = time.time() - countdown_start_time
        remaining = 3 - int(elapsed)
        if remaining > 0:
            cv.putText(debug_image, f"Get Ready: {remaining}", (50, 100), cv.FONT_HERSHEY_DUPLEX, 2, (0, 165, 255), 3)
        else:
            is_counting_down = False
            recording_mode = True

    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            h_img, w_img, _ = debug_image.shape
            x_vals = [lm.x for lm in hand_landmarks]
            y_vals = [lm.y for lm in hand_landmarks]
            brect = [int(min(x_vals)*w_img), int(min(y_vals)*h_img), int(max(x_vals)*w_img), int(max(y_vals)*h_img)]

            index_tip = hand_landmarks[8]
            is_left = (hand_result.handedness[i][0].category_name == "Left") 
            canonical_points = get_canonical_landmarks(hand_landmarks, is_left) 
            normalized_flat = normalize(canonical_points)

            # Record frames
            if recording_mode and current_user:
                batch = augment_landmarks_list(normalized_flat)
                os.makedirs("model", exist_ok=True)
                with open(f"model/{current_user}_data.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in batch:
                        writer.writerow([current_label] + row)
                
                frames_recorded += 1
                cv.putText(debug_image, f"RECORDING: {frames_recorded}/250", (50, 100), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

                if frames_recorded >= 250:
                    recording_mode = False
                    print(f"Finished recording {current_label}. Retraining...")
                    current_hand_model = train_custom_model(current_user)

            # Predict & Execute Actions
            elif current_hand_model and not is_counting_down:
                prediction = current_hand_model.predict([normalized_flat])
                confidence = current_hand_model.predict_proba([normalized_flat])
                pred_label = prediction[0]
                max_conf = np.max(confidence)

                if max_conf > 0.6:
                    cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
                    debug_image = draw_landmarks_on_image(debug_image, hand_result, pred_label, max_conf, brect)
                    
                    if pred_label == "move" and not freeze_cursor:
                        x = np.interp(index_tip.x, (0.25, 0.75), (screen_w, 0))
                        y = np.interp(index_tip.y, (0.25, 0.75), (0, screen_h))
                        curr_screen_x = prev_screen_x + (x - prev_screen_x) / 4
                        curr_screen_y = prev_screen_y + (y - prev_screen_y) / 4
                        curr_screen_x = max(0, min(screen_w - 1, curr_screen_x))
                        curr_screen_y = max(0, min(screen_h - 1, curr_screen_y))
                        mouse.position = (curr_screen_x, curr_screen_y)
                        prev_screen_x, prev_screen_y = curr_screen_x, curr_screen_y
                    
                    elif pred_label == "left_click":
                        if time.time() - last_click_time > click_cooldown:
                            mouse.click(Button.left, 1)
                            last_click_time = time.time()
                            
                    elif pred_label == "scroll_up":
                        mouse.scroll(0, 0.5)
                    elif pred_label == "scroll_down":
                        mouse.scroll(0, -0.5)
                    elif pred_label == "maximize":
                        if time.time() - last_click_time > click_cooldown:
                            keyboard.press(Key.cmd) 
                            keyboard.press(Key.up)
                            keyboard.release(Key.up)
                            keyboard.release(Key.cmd)
                            last_click_time = time.time()
                    elif pred_label=="minimize":
                        if time.time() - last_click_time > click_cooldown:
                            keyboard.press(Key.cmd)
                            keyboard.press(Key.down)
                            # keyboard.press(Key.down)
                            # keyboard.release(Key.down)
                            keyboard.release(Key.down)
                            keyboard.release(Key.cmd)
                            last_click_time = time.time()

    cv.imshow("System", debug_image)

cap.release()
cv.destroyAllWindows()