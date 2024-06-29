# Import necessary libraries
import cv2
import numpy as np

# Load pre-trained face detection and emotion recognition models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model.h5')

# Function to detect face and predict emotion
def detect_face_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)
        
        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label = EMOTIONS[np.argmax(emotion_prediction)]
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

# Load an image for face emotion detection
image = cv2.imread('input_image.jpg')

# Detect face and predict emotion
output_image = detect_face_emotion(image)

# Display the output image
cv2.imshow('Face Emotion Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
