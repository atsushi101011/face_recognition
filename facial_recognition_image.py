import cv2 as cv

#カスケード分類器読み込み
HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

#画像の読み込み
img = cv.imread("./img/IMG_8616.jpg")

#顔検出
face = cascade.detectMultiScale(img)

#顔部を枠で囲む
for x, y, w, h in face:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)

    #顔部のモザイク処理
    # face= img[y:y+h, x:x+w]
    # small_pic = cv.resize(face, (8,8))
    # mosaic = cv.resize(small_pic,(w,h))
    # img[y:y+h, x:x+w]=mosaic

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()