import cv2
import os

xml = "/home/kusw-001/Downloads/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml)

cam = cv2.VideoCapture(0) #use cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640) #width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,400) #hight
face_id=input('\nenter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count=0

while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    img = cv2.flip(img, -1)#flip cam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05, 5)

    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 4)
            count += 1
            cv2.imwrite("/home/kusw-001/project001/fdCam/dataset/User." +str(face_id)+'.'+str(count) +".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('image', img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27: #end for Esc
            break
        elif count >=30:
            break
    
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
