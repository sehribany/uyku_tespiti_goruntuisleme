import cv2
import dlib
from scipy.spatial import distance
from pygame import mixer

mixer.init()
ses = mixer.Sound('alarm.wav')

def calculate_dist(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    dista = (A + B) / (2.0 * C)
    return dista


cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 255), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_dist(leftEye)
        right_ear = calculate_dist(rightEye)

        left_ear = round(left_ear, 2)
        right_ear = round(right_ear, 2)
        if float(right_ear) < 0.23 and float(left_ear) < 0.23:

            ses.play()
            cv2.putText(frame, "Surucu", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 120), 3)
            cv2.putText(frame, "Uyuma", (15, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 120), 3)
        else:
           ses.stop()
           print("Surucu")
           print(str(left_ear) + " " + str(right_ear))

        print("Surucu")
        print(str(left_ear) + " " + str(right_ear))

    cv2.imshow("Uyku Konrtrol 18290006", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()