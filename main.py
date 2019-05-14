import cv2
import numpy as np
from network.image_handler import identify_expression
from AddImages import add_image


def read_emoji_images(emoji_path):
    angry = cv2.imread(emoji_path + "Angry.png", cv2.IMREAD_UNCHANGED)
    disgust = cv2.imread(emoji_path + "Disgust.png", cv2.IMREAD_UNCHANGED)
    fear = cv2.imread(emoji_path + "Fear.png", cv2.IMREAD_UNCHANGED)
    happy = cv2.imread(emoji_path + "Happy.png", cv2.IMREAD_UNCHANGED)
    neutral = cv2.imread(emoji_path + "Neutral.png", cv2.IMREAD_UNCHANGED)
    sad = cv2.imread(emoji_path + "Sad.png", cv2.IMREAD_UNCHANGED)
    surprise = cv2.imread(emoji_path + "Surprise.png", cv2.IMREAD_UNCHANGED)
    return angry, disgust, fear, happy, neutral, sad, surprise

# def handle_face(frame, x, y, w, h):
	# return add_image(frame, angry, (x, y, w, h))

def handle_face(frame, x, y, w, h):
    expression = identify_expression(frame[y:y+h, x:x+w])
    # print(expression)
    if expression == "Angry":
        return add_image(frame, angry, (x, y, w, h))
    elif expression == "Disgust":
        return add_image(frame, disgust, (x, y, w, h))
    elif expression == "Fear":
        return add_image(frame, fear, (x, y, w, h))
    elif expression == "Happy":
        return add_image(frame, happy, (x, y, w, h))
    elif expression == "Neutral":
        return add_image(frame, neutral, (x, y, w, h))
    elif expression == "Sad":
        return add_image(frame, sad, (x, y, w, h))
    elif expression == "Surprise":
        return add_image(frame, surprise, (x, y, w, h))


cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture('sobachk.mp4')
video_writer_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('test_res.mp4', video_writer_fourcc, 20.0, (640, 360))

angry, disgust, fear, happy, neutral, sad, surprise = read_emoji_images("emojis/")

i = 0
while True:
    i += 1
    # if i % 50 != 0:
        # ret, frame = video_capture.read()
        # continue

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        frame = handle_face(frame, x, y, w, h)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
