import hand_detector as hdm
import cv2
import pandas as pd

def main():
    
    signal_data = {}
    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)
        if detector.results.multi_handedness:
            pass
            # print(len(detector.results.multi_handedness))
        #print(detector.)
        all_lmlists = []

        if detector.results.multi_hand_landmarks:
            # Collect landmarks for all detected hands
            for hand_num in range(len(detector.results.multi_handedness)):
                lmlist = detector.find_position(img, draw=False)
                if lmlist:
                    all_lmlists.append(lmlist)

        for lmlist in all_lmlists:
            if lmlist:

                p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25, min(lmlist[x][2] for x in range(len(lmlist))) - 25)
                p4 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25, max(lmlist[x][2] for x in range(len(lmlist))) + 25)
                cv2.rectangle(img, p1, p4, (255,255,255), 3, 1)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and lmlist:
            print(len(all_lmlists))
            print(all_lmlists)
            print()
            '''for item in lmlist:
                if f'{item[0]}x' in signal_data:
                    signal_data[f'{item[0]}x'].append(item[1])
                else:
                    signal_data[f'{item[0]}x'] = [item[1]]
                if f'{item[0]}y' in signal_data:
                    signal_data[f'{item[0]}y'].append(item[2])
                else:
                    signal_data[f'{item[0]}y'] = [item[2]]'''
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
data.to_csv('hand_signals.csv')'''
