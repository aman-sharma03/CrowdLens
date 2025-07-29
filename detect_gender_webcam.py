import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


model = load_model('gender_detection.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


count_men = 0
count_women = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    count_men = 0
    count_women = 0
    
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        face = cv2.resize(face, (96, 96))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0) / 255.0
 
        prediction = model.predict(face)
        gender = 'Woman' if prediction[0][1] > 0.5 else 'Man'
        
        if gender == 'Woman':
            count_women += 1
        else:
            count_men += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    cv2.putText(frame, f"Men: {count_men}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Women: {count_women}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Gender Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
