import csv
import copy
import itertools
from collections import Counter, deque
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk  # Importing from PIL

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        
        # Initialize mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.mode = 0
        self.cap = None
        
        # Tkinter GUI setup
        self.video_label = ttk.Label(self.root)
        self.video_label.pack()

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_video)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT)

        self.quit_button = ttk.Button(self.root, text="Quit", command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT)

    def start_video(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
        self.process_video()

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_label.config(image='')

    def quit_app(self):
        self.stop_video()
        self.root.quit()

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            messagebox.showerror("Error", "Failed to capture video")
            return

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(debug_image, self.point_history)
                self.logging_csv(-1, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                debug_image = self.draw_bounding_rect(True, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id], self.point_history_classifier_labels[most_common_fg_id[0][0]])

        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_info(debug_image, self.cvFpsCalc.get(), self.mode, -1)

        self.display_image(debug_image)
        self.root.after(10, self.process_video)

    def display_image(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (960, 540))
        img = Image.fromarray(img)  # Convert to PIL Image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert to ImageTk PhotoImage
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    # Utility functions
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
            cv.rectangle(image, (brect[0], brect[1] - 22), (brect[0] + 50, brect[1]), (0, 0, 0), -1)
        return image

    def draw_landmarks(self, image, landmark_point):
        for point in landmark_point:
            cv.circle(image, tuple(point), 5, (255, 0, 0), -1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        info_text = handedness.classification[0].label[0:] + ': ' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image

    def draw_point_history(self, image, point_history):
        for point in point_history:
            cv.circle(image, tuple(point), 1, (255, 0, 0), -1)
        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, "MODE:" + str(mode), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, "NUMBER:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        return image


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
