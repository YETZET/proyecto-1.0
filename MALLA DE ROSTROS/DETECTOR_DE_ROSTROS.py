import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 
        scaleFactor=1.7,
        minNeighbors=5,
        minSize=(200,200),
        maxSize=(150,150))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        #cv2.imshow('frame',frame)
        
            imageOut =frame[y:y+h,x:x+w]
            height, width,_=imageOut.shape
            imageOut_rgb = cv2.cvtColor(imageOut, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(imageOut_rgb)
            #print("Face landmarks: ", results.multi_face_landmarks)
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(imageOut, face_landmarks,
                        mp_face_mesh.FACE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))


                    x1 = int(face_landmarks.landmark[0].x * width)
                    y1 = int(face_landmarks.landmark[0].y * height)
                    x2 = int(face_landmarks.landmark[1].x * width)
                    y2 = int(face_landmarks.landmark[1].y * height)
                    d1=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[1].x * width)
                    y1 = int(face_landmarks.landmark[1].y * height)
                    x2 = int(face_landmarks.landmark[2].x * width)
                    y2 = int(face_landmarks.landmark[2].y * height)
                    d2=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[2].x * width)
                    y1 = int(face_landmarks.landmark[2].y * height)
                    x2 = int(face_landmarks.landmark[3].x * width)
                    y2 = int(face_landmarks.landmark[3].y * height)
                    d3=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[3].x * width)
                    y1 = int(face_landmarks.landmark[3].y * height)
                    x2 = int(face_landmarks.landmark[4].x * width)
                    y2 = int(face_landmarks.landmark[4].y * height)
                    d4=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[4].x * width)
                    y1 = int(face_landmarks.landmark[4].y * height)
                    x2 = int(face_landmarks.landmark[5].x * width)
                    y2 = int(face_landmarks.landmark[5].y * height)
                    d5=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[5].x * width)
                    y1 = int(face_landmarks.landmark[5].y * height)
                    x2 = int(face_landmarks.landmark[6].x * width)
                    y2 = int(face_landmarks.landmark[6].y * height)
                    d6=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[6].x * width)
                    y1 = int(face_landmarks.landmark[6].y * height)
                    x2 = int(face_landmarks.landmark[7].x * width)
                    y2 = int(face_landmarks.landmark[7].y * height)
                    d7=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[7].x * width)
                    y1 = int(face_landmarks.landmark[7].y * height)
                    x2 = int(face_landmarks.landmark[8].x * width)
                    y2 = int(face_landmarks.landmark[8].y * height)
                    d8=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[8].x * width)
                    y1 = int(face_landmarks.landmark[8].y * height)
                    x2 = int(face_landmarks.landmark[9].x * width)
                    y2 = int(face_landmarks.landmark[9].y * height)
                    d9=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[9].x * width)
                    y1 = int(face_landmarks.landmark[9].y * height)
                    x2 = int(face_landmarks.landmark[10].x * width)
                    y2 = int(face_landmarks.landmark[10].y * height)
                    d10=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[10].x * width)
                    y1 = int(face_landmarks.landmark[10].y * height)
                    x2 = int(face_landmarks.landmark[11].x * width)
                    y2 = int(face_landmarks.landmark[11].y * height)
                    d11=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[11].x * width)
                    y1 = int(face_landmarks.landmark[11].y * height)
                    x2 = int(face_landmarks.landmark[12].x * width)
                    y2 = int(face_landmarks.landmark[12].y * height)
                    d12=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[12].x * width)
                    y1 = int(face_landmarks.landmark[12].y * height)
                    x2 = int(face_landmarks.landmark[13].x * width)
                    y2 = int(face_landmarks.landmark[13].y * height)
                    d13=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[13].x * width)
                    y1 = int(face_landmarks.landmark[13].y * height)
                    x2 = int(face_landmarks.landmark[14].x * width)
                    y2 = int(face_landmarks.landmark[14].y * height)
                    d14=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[14].x * width)
                    y1 = int(face_landmarks.landmark[14].y * height)
                    x2 = int(face_landmarks.landmark[15].x * width)
                    y2 = int(face_landmarks.landmark[15].y * height)
                    d15=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[15].x * width)
                    y1 = int(face_landmarks.landmark[15].y * height)
                    x2 = int(face_landmarks.landmark[16].x * width)
                    y2 = int(face_landmarks.landmark[16].y * height)
                    d16=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[16].x * width)
                    y1 = int(face_landmarks.landmark[16].y * height)
                    x2 = int(face_landmarks.landmark[17].x * width)
                    y2 = int(face_landmarks.landmark[17].y * height)
                    d17=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[17].x * width)
                    y1 = int(face_landmarks.landmark[17].y * height)
                    x2 = int(face_landmarks.landmark[18].x * width)
                    y2 = int(face_landmarks.landmark[18].y * height)
                    d18=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[18].x * width)
                    y1 = int(face_landmarks.landmark[18].y * height)
                    x2 = int(face_landmarks.landmark[19].x * width)
                    y2 = int(face_landmarks.landmark[19].y * height)
                    d19=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[19].x * width)
                    y1 = int(face_landmarks.landmark[19].y * height)
                    x2 = int(face_landmarks.landmark[20].x * width)
                    y2 = int(face_landmarks.landmark[20].y * height)
                    d20=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[20].x * width)
                    y1 = int(face_landmarks.landmark[20].y * height)
                    x2 = int(face_landmarks.landmark[21].x * width)
                    y2 = int(face_landmarks.landmark[21].y * height)
                    d21=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[21].x * width)
                    y1 = int(face_landmarks.landmark[21].y * height)
                    x2 = int(face_landmarks.landmark[22].x * width)
                    y2 = int(face_landmarks.landmark[22].y * height)
                    d22=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[22].x * width)
                    y1 = int(face_landmarks.landmark[22].y * height)
                    x2 = int(face_landmarks.landmark[23].x * width)
                    y2 = int(face_landmarks.landmark[23].y * height)
                    d23=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[23].x * width)
                    y1 = int(face_landmarks.landmark[23].y * height)
                    x2 = int(face_landmarks.landmark[24].x * width)
                    y2 = int(face_landmarks.landmark[24].y * height)
                    d24=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[24].x * width)
                    y1 = int(face_landmarks.landmark[24].y * height)
                    x2 = int(face_landmarks.landmark[25].x * width)
                    y2 = int(face_landmarks.landmark[25].y * height)
                    d25=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    x1 = int(face_landmarks.landmark[25].x * width)
                    y1 = int(face_landmarks.landmark[25].y * height)
                    x2 = int(face_landmarks.landmark[26].x * width)
                    y2 = int(face_landmarks.landmark[26].y * height)
                    d26=math.sqrt((x2-x1)**2+(y2-y1)**2)

                    disMedida=d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15+d16+d17+d18+d19+d20+d21+d22+d23+d24+d25+d26
                    disNomMax=754
                    disNomMin=554

                   
                    valorDesMax=disNomMax-disMedida
                    valorDesMin=disMedida-disNomMin

                    print("Distance+: ", valorDesMax)
                    print("Distance-: ", valorDesMin)
               
                    
    

            
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

cap.release()
cv2.destroyAllWindows()
                 