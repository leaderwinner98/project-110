import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('keras_model.h5')


camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    status, frame = camera.read()
    if status:
        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (224, 224))
        expanded_frame = np.expand_dims(resized_frame, axis=0)
        normalized_frame = expanded_frame / 255.0
        predictions = model.predict(normalized_frame)
        gesture_class = np.argmax(predictions)
        gestures = ['Rock', 'Paper', 'Scissors']
        predicted_gesture = gestures[gesture_class]
        cv2.putText(frame, predicted_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('feed', frame)
        code = cv2.waitKey(1)

        if code == 32:
            break

camera.release()


cv2.destroyAllWindows()
