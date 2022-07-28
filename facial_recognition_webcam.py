import cv2 as cv
import numpy as np

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

cap = cv.VideoCapture(0)

anime_face = cv.imread('./img/face_icon.png')

def anime_face_func(img, rect):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    img_face = cv.resize(anime_face, (w, h))
 
    img2 = img.copy()
    img2[y1:y2, x1:x2] = img_face
    return img2

while(True):
    ret, frame = cap.read()

    face = cascade.detectMultiScale(frame)

    for x, y, w, h in face:
        frame = anime_face_func(frame, (x,y,x+w,y+h))
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

        # faceImg = cv.resize(faceImg, ((int)((x+w)*1.3), (int)((y+h)*1.3)), cv.IMREAD_UNCHANGED)
        # x -= (x+w)*0.15  # x_offset
        # y -= (y+h)*0.15  # y_offset

        # if len(face) > 0:
        #         for rect in face:
        #             count += 1
        #             if count < 4:
        #                 aveSize += (rect[2]+rect[3])/2
        #                 break
        #             elif count == 4:
        #                 aveSize += (rect[2]+rect[3])/2
        #                 aveSize /= 5
        #             else:
        #                 aveSize = aveSize*0.8+rect[2]*0.1+rect[3]*0.1
        #             thresh = aveSize*0.95  # 移動平均の95%以上を閾値
        #             if rect[2] < thresh or rect[3] < thresh:
        #                 break
        #             # 検出した顔を囲む矩形の作成
        #             #cv2.rectangle(frame, tuple(rect[0:2]), tuple( rect[0:2]+rect[2:4]), color, thickness=2)
        #             faceImg = cv.resize(
        #                 faceImg, ((int)(rect[2]*1.3), (int)(rect[3]*1.3)), cv.IMREAD_UNCHANGED)

        #             rect[0] -= rect[2]*0.15  # x_offset
        #             rect[1] -= rect[3]*0.15  # y_offset
        #             frame[rect[1]:rect[1]+faceImg.shape[0],
        #                   rect[0]:rect[0]+faceImg.shape[1]] = frame[rect[1]:rect[1]+faceImg.shape[0],
        #                                                             rect[0]:rect[0]+faceImg.shape[1]] * (1 - faceImg[:, :, 3:] / 255) + \
        #                 faceImg[:, :, :3] * (faceImg[:, :, 3:] / 255)

        # #モザイク処理
        # face= frame[y:y+h, x:x+w]
        # small_pic = cv.resize(face, (5,5))
        # mosaic = cv.resize(small_pic,(w,h))
        # frame[y:y+h, x:x+w]=mosaic

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()