import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("trainer\\trainner.yml")

id=0

fontface=cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id1,conf=rec.predict(gray[y:y+h,x:x+w])
        print(id1)
        if(id1==1):
            id1="venkat"
        if(id1==2):
            id1 ="harishma"
        if(id1 ==3):
            id1 = "sravanthi"
          
        cv2.putText(img,str(id1),(x,y+h),fontface,2,(255,0,0),3);
    cv2.imshow("Face",img);   
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
