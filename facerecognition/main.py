import face_recognition
import cmake
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Loading known faces
# Ecoding - Coverting images to number so that it becomes easier to compare
shambhavi_image = face_recognition.load_image_file("faces/shambhavi.jpg")
# Pehle face chahiye issliye zero ([0]) likh diya
shambhavi_encoding = face_recognition.face_encodings(shambhavi_image)[0]

shaurya_image = face_recognition.load_image_file("faces/shaurya.jpg")
shaurya_encoding = face_recognition.face_encodings(shaurya_image)[0]

know_face_encodings = [shambhavi_encoding, shaurya_encoding]
know_face_names = ["Shambhavi", "Shaurya"]

# List of expected students

students = know_face_names.copy()

#location of face in the image
face_locations = []
face_encodings = []

#Get the current date and time

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(know_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = know_face_names[best_match_index]

        #Add a text if a person is present
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1.5
        fontColor = (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, name+" Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
