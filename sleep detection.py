import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

# Let's the initialize the audio mixer from pygame and load an audio file that we used as an alarm.
mixer.init()
alarm = mixer.Sound('wake-up.wav')
# Let's load the haar cascade classifier for detecting face, left eye and right eye.
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
lefteye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
righteye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
# Let's define a label for desciribing the eye states. when the eye become close the value will be 0 & the value will be 1 when the eye opens.
label=['Close', 'Open']
# Let's Load Our Pre-Trained Model
model = load_model('models/cnnSleep.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
count=0
score=0
thickness=2
rightEyePrediction=[99]
leftEyePrediction=[99]

while(True):
    # Let's start capturing frames from the videos
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    # Let's the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Let's detect face, left eye and right eye from the frame by using the haar cascade classifier.
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = lefteye.detectMultiScale(gray)
    right_eye = righteye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100,100,100), 1)

    for (x, y, w, h) in right_eye:
        # Let's Extract ROI of right eye
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        # Reshape the ROI
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        # Let's use the model to predict the eyes state
        rightEyePrediction = model.predict(r_eye)
        if np.argmax(rightEyePrediction) == 1:
            label = 'Open'
        else:
            label = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        leftEyePrediction = model.predict(l_eye)
        if np.argmax(leftEyePrediction) == 1:
            label = 'Open'
        else:
            label = 'Closed'
        break

    if np.argmax(rightEyePrediction) == 0 and np.argmax(leftEyePrediction) == 0:
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_8)
    else:
        score = score-1
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_8)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_8)
    if score > 10:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            alarm.play()
        except:
            pass
        if thickness < 16:
            thickness = thickness + 2
        else:
            thickness = thickness - 2
            if thickness < 2:
                thickness = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()