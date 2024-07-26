import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=.5, presence_con=.5, track_con=.5) -> None:
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.presence_con = presence_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw =  mp.solutions.drawing_utils


    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        #if self.results.multi_handedness:
        #    print(len(self.results.multi_handedness))

        return img
    
    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_num in range(len(self.results.multi_handedness)):
                hand = self.results.multi_hand_landmarks[hand_num]
                for id, lm in enumerate(hand.landmark):
                    #print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lm_list.append([id, cx, cy])
                    if id in [4, 8, 12, 16, 20] and draw:
                    #if draw:
                        cv2.circle(img, (cx,cy), 10, (255,8,255), cv2.FILLED)

        return lm_list

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)
        if lmlist:
            print(lmlist[8])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()