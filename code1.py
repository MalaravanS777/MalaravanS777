from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
import time
import threading
import queue

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert_sound():
    pygame.mixer.init()
    alert_sound_file = r"E:\Malaravan vscode\mini project\mixkit-happy-bells-notification-937.wav"  # Path to your alert sound file
    pygame.mixer.music.load(alert_sound_file)
    pygame.mixer.music.play()
    time.sleep(5)  # Play the sound for 5 seconds

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"E:\Malaravan vscode\mini project\Drowsiness_Detection-master\models\shape_predictor_68_face_landmarks.dat")  # Update the path to your shape predictor

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
alert_timer = None
text_queue = queue.Queue()

def capture_frames(alert_timer):
    global flag
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check and alert_timer is None:
                    text_queue.put("****************ALERT!****************")  # Put the text in the queue
                    play_alert_sound()  # Play the alert sound for 5 seconds
                    alert_timer = time.time()
                    flag = 0  # Reset the flag to prevent continuous alerts

        cv2.imshow("Frame", frame)

        # Capture key input
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    # Release resources
    cv2.destroyAllWindows()
    cap.release()

# Create a separate thread for capturing frames
frame_thread = threading.Thread(target=capture_frames, args=(alert_timer,))
frame_thread.start()

# Main thread continues
while True:
    try:
        text = text_queue.get_nowait()
        print(text)  # Print the text from the queue
    except queue.Empty:
        pass

    if frame_thread.is_alive():
        continue
    else:
        break
