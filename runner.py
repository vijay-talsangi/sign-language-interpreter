import hand_detector as hdm
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import time
from gtts import gTTS
import io
import pygame

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def speech(text):
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    myobj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Load the BytesIO object as a sound
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()

    # Keep the program running while the sound plays
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
    data = pd.read_csv('hand_signals.csv')
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)
    X_train = train.drop('letter', axis=1)
    y_train = train['letter']
    X_test = test.drop('letter', axis=1)
    y_test = test['letter']

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    pygame.mixer.init()
    signal_data = {}
    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()
    letters = [0]
    word = ''
    words = []
    start = time.time()
    end = time.time()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=False)
        lmlist = detector.find_position(img)
        confidence_threshold = .7
        
        key = cv2.waitKey(1) & 0xFF
        #print(len(lmlist) == 0)
        if not lmlist:
            end = time.time()
            idle_timer = end-start
            if idle_timer >= 3 and word != '':
                if word[-1] != ' ':
                    myobj = gTTS(text=word, lang='en', slow=False)
                    mp3_fp = io.BytesIO()
                    myobj.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)

                    # Load the BytesIO object as a sound
                    pygame.mixer.music.load(mp3_fp, 'mp3')
                    pygame.mixer.music.play()

                    # Keep the program running while the sound plays
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    words.append(word)
                    word =word + ' '

        
        
        
        
        
        
        
            #print(end-start)
            #print(False)
        if lmlist and len(lmlist) == 21:
            start = time.time()
            
            p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25, min(lmlist[x][2] for x in range(len(lmlist))) - 25)
            p2 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25, max(lmlist[x][2] for x in range(len(lmlist))) + 25)
            cv2.rectangle(img, p1, p2, (255,255,255), 3)

            location_vector = np.array([coord for lm in lmlist for coord in lm[1:3]]).reshape(1, -1)
            
            probabilities = model.predict_proba(location_vector)
            max_prob = np.max(probabilities)
            if max_prob > confidence_threshold:
                predicted_letter = model.predict(location_vector)[0]
                if predicted_letter == letters[-1]:
                    letters.append(predicted_letter)
                else:
                    letters = [predicted_letter]
                cv2.putText(img, predicted_letter, (p1[0], p1[1] - 10), cv2.QT_FONT_NORMAL, 3, (255, 255, 255), 3)
            
            if len(letters) == 20:
                word = word + letters[0]
                letters = [0]
                print(word)
            
            
            #print(letters)
            #cv2.putText(img, model.predict(location_vector)[0], (p1[0], p1[1]-10), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 3)
            #print(model.predict_proba(location_vector))
            #p3 = (p1[0])
            #cv2.rectangle(img, p1, p4, (255,255,255), cv2.FILLED)

        
        
        
        if key == ord('c') and lmlist:
            for item in lmlist:
                if f'{item[0]}x' in signal_data:
                    signal_data[f'{item[0]}x'].append(item[1])
                else:
                    signal_data[f'{item[0]}x'] = [item[1]]
                if f'{item[0]}y' in signal_data:
                    signal_data[f'{item[0]}y'].append(item[2])
                else:
                    signal_data[f'{item[0]}y'] = [item[2]]
        cv2.imshow("Image", img)
        if key == ord('q'):
            break
        #cv2.waitKey(1)
    if signal_data:
        signal_data['letter'] = ['a'] * len(signal_data['0x'])
        new_signals = pd.DataFrame(signal_data)
        existing_signals = pd.read_csv('hand_signals.csv')
        updated_stats = pd.concat([existing_signals, new_signals], ignore_index=True)
        updated_stats.to_csv('hand_signals.csv', index=False)

if __name__ == '__main__':
    main()

'''dict = {}
lst = [[0, 852, 534], [1, 802, 481], [2, 778, 409], [3, 761, 350], [4, 734, 300], [5, 849, 335], [6, 853, 262], [7, 852, 220], [8, 848, 180], [9, 887, 346], [10, 893, 272], [11, 880, 228], [12, 860, 191], [13, 921, 372], [14, 945, 309], [15, 931, 278], [16, 903, 256], [17, 952, 412], [18, 975, 362], [19, 965, 341], [20, 940, 329]]
print()
for item in lst:
    dict[f'{item[0]}x'] = []
    dict[f'{item[0]}y'] = []

data = pd.DataFrame(dict)
data.to_csv('hand_signals.csv')
'''

'''data = pd.read_csv('hand_signals.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)
X_train = train.drop('letter', axis=1)
y_train = train['letter']
X_test = test.drop('letter', axis=1)
y_test = test['letter']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)'''

'''data = pd.read_csv('hand_signals.csv')
data = data.drop(data[data['letter'] == 'c'].index)

data.to_csv('hand_signals.csv')
'''