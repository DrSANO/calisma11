from ultralytics import YOLO
import cv2 as cv
import numpy as np
import time

model = YOLO("fwlastn.pt")
cam=cv.VideoCapture("hareketli_ucak3.mp4")
secc=0
while True:
    lastsec=time.time()
    ret, gor = cam.read()
    gor= cv.resize(gor, (1280,720))
    sonuc= model(gor,stream=True)
    for s in sonuc:
        boxes=s.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv.rectangle(gor,(x1,y1),(x2,y2),(255,0,0),3)
            """conf=box.conf[0]
            conf=round(conf,2)"""

    fps=1/(lastsec-secc)
    secc=lastsec
    print("fps:",fps)
    cv.imshow("sa",gor)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv.destroyAllWindows()
