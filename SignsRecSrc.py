import numpy as np
import cv2
#face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
img=cv2.imread('sign1.jfif')
nimg=cv2.imread('sign2.jfif')
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, baw) = cv2.threshold(gimg, 127, 255, cv2.THRESH_BINARY)
hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsvimg2 = cv2.cvtColor(nimg, cv2.COLOR_BGR2HSV)

# Red color
low_red = np.array([161, 155, 55])
high_red = np.array([179, 255, 255])
red_mask = cv2.inRange(hsvimg, low_red, high_red)
red_mask2 = cv2.inRange(hsvimg2, low_red, high_red)
red1 = cv2.bitwise_and(img,img, mask=red_mask)
red2 = cv2.bitwise_and(nimg,nimg, mask=red_mask2)
orb = cv2.ORB_create(nfeatures=2000)
keypoints, descriptors = orb.detectAndCompute(img,red_mask)
keypoints2, descriptors2 = orb.detectAndCompute(nimg,red_mask2)

re1=cv2.drawKeypoints(img, keypoints,None)
re2=cv2.drawKeypoints(nimg, keypoints2,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches=bf.match(descriptors,descriptors2)
print(len(matches))

matches=sorted(matches,key = lambda x:x.distance)

result=cv2.drawMatches(img,keypoints,nimg,keypoints2,matches[:10],None)

#cv2.imshow('baw',baw)
#cv2.imshow('sign.jpg',img)
#cv2.imshow('mask',red_mask )
#cv2.imshow('red',red)
#cv2.imshow("ORB kp",img2)
cv2.imshow('re1',re1)
cv2.imshow('re2',re2)

cv2.imshow("result",result)

cv2.waitKey(0)
cv2.destroyAllWindows()