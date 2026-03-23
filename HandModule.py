import cv2
import mediapipe as mp
import math


class handDetector():
    def __init__(self, mode=False, maxHands=4, detectionCon=0.5, trackCon=0.5):
        self.mode        = mode
        self.maxHands    = maxHands
        self.detectionCon= detectionCon
        self.trackCon    = trackCon
        self.mpHands     = mp.solutions.hands
        self.hands       = self.mpHands.Hands(
            static_image_mode       = self.mode,
            max_num_hands           = self.maxHands,
            min_detection_confidence= self.detectionCon,
            min_tracking_confidence = self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
        return img

    def findPosition(self, img, handNo=-1, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            if handNo < 0:
                for i, myhand in enumerate(self.results.multi_hand_landmarks):
                    for id, lm in enumerate(myhand.landmark):
                        h, w, c = img.shape
                        cx, cy  = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id, cx, cy])
                        if draw and id == 0:
                            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            else:
                if self.results.multi_hand_landmarks:
                    myhand = self.results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myhand.landmark):
                        h, w, c = img.shape
                        cx, cy  = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id, cx, cy])
                        if draw and id == 0:
                            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        return lmlist


class doublehandDetector():
    def __init__(self, mode=False, maxHands=4, detectionCon=0.5, trackCon=0.5):
        self.mode         = mode
        self.maxHands     = maxHands
        self.detectionCon = detectionCon
        self.trackCon     = trackCon
        self.mpHands      = mp.solutions.hands
        self.hands        = self.mpHands.Hands(
            static_image_mode       = self.mode,
            max_num_hands           = self.maxHands,
            min_detection_confidence= self.detectionCon,
            min_tracking_confidence = self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
        return img

    def findPositionleft(self, img, draw=True):
        leftlist = []
        if self.results.multi_hand_landmarks:
            for i, myhand in enumerate(self.results.multi_hand_landmarks):
                hand_label = self.results.multi_handedness[i].classification[0].label
                if hand_label == 'Right':  # mirrored camera
                    for id, lm in enumerate(myhand.landmark):
                        h, w, c = img.shape
                        cx, cy  = int(lm.x * w), int(lm.y * h)
                        leftlist.append([id, cx, cy])
                        if draw and id == 0:
                            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        return leftlist

    def findPositionright(self, img, draw=True):
        rightlist = []
        if self.results.multi_hand_landmarks:
            for i, myhand in enumerate(self.results.multi_hand_landmarks):
                hand_label = self.results.multi_handedness[i].classification[0].label
                if hand_label == 'Left':  # mirrored camera
                    for id, lm in enumerate(myhand.landmark):
                        h, w, c = img.shape
                        cx, cy  = int(lm.x * w), int(lm.y * h)
                        rightlist.append([id, cx, cy])
                        if draw and id == 0:
                            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        return rightlist
