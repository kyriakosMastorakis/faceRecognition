import cv2
from datetime import datetime

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
recColor = (255,255,255)

def detect_bouncing_box(vid):
    BlackWhite = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
   # BlackWhite = cv2.flip(BlackWhite, 1)
    faces = face_classifier.detectMultiScale(BlackWhite, 1.1, 5, minSize=(40, 40))
    #print("Faces:", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (recColor), 2)
    return faces


while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = detect_bouncing_box(frame)


    face_count = len(faces)
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if  key % 256 == 27:    #esc button
        break
    elif key % 256 == ord('r'):  #red
        recColor = (0,0,255)
    elif key % 256 == ord('b'): #blue
        recColor = (255,0,0)
    elif key % 256 == ord('y'):  #yellow
        recColor = (0,204,255)
    elif key % 256 == ord('g'):  #green
        recColor = (0,255,0)
    elif key % 256 == ord('p'):  #purple
        recColor = (191,0,191)
    elif key % 256 == ord(' '):
        now = datetime.now()
        ImgName = "Frame_at_time" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second) + ".png"
        cv2.imwrite(ImgName, frame)
        print("image saved: " + ImgName)


cap.release()
cv2.destroyAllWindows()

