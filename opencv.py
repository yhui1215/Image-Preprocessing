
# *error* pip install opencv-python 설치 후
# --> ImportError: numpy.core.multiarray failed to import
# -->디폴트로 설치되는 numpy 1.19.4 대신에 numpy 1.19.3

# *error* 출력 안되고 Process finished with exit code 0
# --> run edit 가서 run with python console

'''
import os

path=F'C:/Users/admin/PycharmProjects/'
os.chdir(path)

print("package imported")
'''
#### <Read Images-Video>

# < step 1 >
'''
import cv2
img=cv2.imread(r"cat2.jpg")
cv2.imshow("output",img)
cv2.waitKey(0)
'''
# *error* error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'cv::imshow'
# --> 전체 경로를 작성해야한다. / 또는 \\, //는 아니다
# --> 파일명 영어로 변환하기# --> png 이미지를 사용하는 경우 pypng 모듈을 설치? X
# --> os.chdir

# < step 2 >
'''
import cv2
cap=cv2.VideoCapture("family.mp4")
while True:
    success,img=cap.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break;
'''
# https://copycoding.tistory.com/154

# < step 3 >
'''
import cv2
cap=cv2.VideoCapture(0)
#파라미터 설정
cap.set(3,640) # 가로길이
cap.set(4,480) # 세로길이
while True:
    # 한 프레임씩 읽기
    success,img=cap.read()
    # 비디오 실행
    cv2.imshow("Video",img)
    # 1초 후 실행키를
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break;
# cap의 객체를 종료하고,
cap.release()
# 화면도 끄기기
cv2.destroyAllWindows()
'''



# < step 4 >
'''
import cv2
import numpy as np
img=cv2.imread("purple.png")
kernel=np.ones((5,5),np.uint8)
#print(kernel)

imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgblur=cv2.GaussianBlur(imggray,(5,5),0)
imgcanny1=cv2.Canny(img,50,50)
imgcanny=cv2.Canny(img,100,100)
imgdialation=cv2.dilate(imgcanny,kernel,iterations=2)
imgeroded=cv2.erode(imgdialation,kernel,iterations=1)

# cv2.imshow("gray image!",imggray)
# cv2.imshow("blur image!",imgblur)
# cv2.imshow("canny image!",imgcanny)
cv2.imshow("canny image!",imgcanny)
cv2.imshow("dialation image!",imgdialation)
cv2.imshow("eroded image!",imgeroded)
cv2.waitKey(0)
'''


# < step 5 >
'''
img=cv2.imread("cat2.jpg")
print(img.shape)

imgresize=cv2.resize(img,(100,150))
print(imgresize.shape)

imgcropped=img[0:150,0:150]
cv2.imshow("image",img)
cv2.imshow("image resize",imgresize)
cv2.imshow("image cropped",imgcropped)
cv2.waitKey(0)
'''
# < step 6 >
'''
import cv2
import numpy as np
img=np.zeros((512,512,3),np.uint8)
print(img.shape)
#img[:]=255,0,0
img[200:300,100:300]=255,0,0
#cv2.line(img,(0,0),(300,300),(0,255,0),3)
#cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
#cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
#cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)
#cv2.circle(img,(400,50),30,(255,255,0),5)
#cv2.putText(img," opencv ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
cv2.imshow("image",img)
cv2.waitKey(0)
'''
# < step 7 >

'''
import cv2
import numpy as np
width=200
height=200
img=cv2.imread("cat2.jpg")
pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgOutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("image",imgOutput)
cv2.waitKey(0)
'''
'''
# < step 8 >
import cv2
import numpy as np
img=cv2.imread("cat2.jpg")
purple=cv2.imread("purple.png")
hor = np.hstack((img,img,img))
ver = np.vstack((img,img))
cv2.imshow("horizontal",hor)
cv2.imshow("vertical",ver)
cv2.waitKey(0)
'''
import cv2

import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

"""
img=cv2.imread("purple.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgStack = stackImages(0.5, ([img, imgGray, imgGray], [img, img, img]))
cv2.imshow("ImageStack", imgStack)
cv2.waitKey(0)
"""


# < step 9 >
import cv2
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 30, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 162, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 200, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 165, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread("purple.png")
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Original",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)