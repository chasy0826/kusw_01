import cv2
import numpy as np
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
xml = "/home/kusw-001/Downloads/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml)
font=cv2.FONT_HERSHEY_SIMPLEX
id=0
names=['csy']

cap = cv2.VideoCapture(0) #use cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400) #hight

minW=0.1*cap.get(cv2.CAP_PROP_FRAME_WIDTH)
minH=0.1*cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
            id, confidence=recognizer.predict(gray[y:y+h, x:x+w])
            if(onfidence<100):
                id=names[id]
                confidence=" {0}%".format(round(100-confidence))
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
                id="unknown"
                confidence=" {0}%".fomat(round(100-confidence))
            cv2.putText(img, str(id), (x+5, y-5), font, 1, (0, 255, 0), 4)
            cv2.putText(str(confidence), (x+5, y+h-5), font, 1, (0, 255, 0), 2)
            
    else:
        cv2.imshow("result", frame)
        out.write(frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27: #end for Esc
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
