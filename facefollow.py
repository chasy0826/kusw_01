import numpy as np
import cv2

xml = "/home/kusw-001/Downloads/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml)

cap = cv2.VideoCapture(0) #use cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400) #hight

out=cv2.VideoWriter("/home/kusw-001/Videos/output.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, (640,400))

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, -1)#flip cam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05, 5)
    print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 4)
            yl=int(h/2-100)
            xl=int(w/2-160)
            yr=int(h/2+120)
            xr=int(h/2+160)
            face_img=frame[y+yl:y+yr,x+xl:x+xr]
            try:
                face_img=cv2.resize(face_img, (640,400), interpolation=cv2.INTER_AREA)
                cv2.imshow("result", face_img)
                out.write(face_img)
                
            except:
                cv2.imshow("result", frame)
                out.write(frame)
                        
            
    else:
        cv2.imshow("result", frame)
        out.write(frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27: #end for Esc
        break

cap.release()
cv2.destroyAllWindows()
