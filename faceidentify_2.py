import cv2
import numpy as np
from PIL import Image
import os

path='/home/kusw-001/project001/fdCam/dataset'
recognizer=cv2.face.LBPHFaceRecognizer_create()
xml = "/home/kusw-001/Downloads/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]
    for imagePath in imagePaths:
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img, 'uint8')
        id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_numpy)
        for(x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids
print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids=getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('triner/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
