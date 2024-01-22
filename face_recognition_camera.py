import cv2

cap = cv2.VideoCapture(0)
face_recognition = cv2.face.LBPHFaceRecognizer.create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognition.read('lbph_classifier.yml')
names = ["Jones", "Grabiel", "Vu", "Giau", "John", "Kevin", "Ari", "Adam", "Meyer", "Alex", "Xander", "Hexan"]
while True:
    ret, face = cap.read()
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(gray_face, scaleFactor=1.5, minSize=(30, 30))
    for (x, y, w, h) in detections:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(face, top_left, bottom_right, (0, 255, 0), 2)
        img_face = cv2.resize(gray_face[y:y+w, x:x+h], (220, 220))
        idx_predict = face_recognition.predict(img_face)[0]
        name = names[idx_predict - 1]
        cv2.putText(face, name, (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))

    cv2.imshow('Image', face)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
