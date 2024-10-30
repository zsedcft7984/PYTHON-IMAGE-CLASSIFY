from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
#np.set_printoptions(suppress=True)

# Load the model
model = load_model("bottel&mentos.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
if camera.isOpened():
    delay=int(10/camera.get(cv2.CAP_PROP_FPS)) #한프레임당 걸리는 시간 계산하기
    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()

        frame = image[:, :, :] 
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]



        cv2.imshow("Webcam Image", frame)

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break
else:
    print("비디오를 읽어들일 수 없습니다.")
camera.release()
cv2.destroyAllWindows()
