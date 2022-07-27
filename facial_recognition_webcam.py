import cv2 as cv

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    face = cascade.detectMultiScale(frame)

    for x, y, w, h in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

        # #モザイク処理
        # face= frame[y:y+h, x:x+w]
        # small_pic = cv.resize(face, (8,8))
        # mosaic = cv.resize(small_pic,(w,h))
        # frame[y:y+h, x:x+w]=mosaic

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()