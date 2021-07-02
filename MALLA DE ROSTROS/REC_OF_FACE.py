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

             # DETECCION DE OJOS
                    x1 = int(face_landmarks.landmark[33].x * width)
                    y1 = int(face_landmarks.landmark[33].y * height)
                    x2 = int(face_landmarks.landmark[133].x * width)
                    y2 = int(face_landmarks.landmark[133].y * height)
                    d1=math.sqrt((x2-x1)**2+(y2-y1)**2)
                    
             # DETECCION DE LAGRIMAL
                    x3 = int(face_landmarks.landmark[133].x * width)
                    y3 = int(face_landmarks.landmark[133].y * height)
                    x4 = int(face_landmarks.landmark[243].x * width)
                    y4 = int(face_landmarks.landmark[243].y * height)
                    d2=math.sqrt((x4-x3)**2+(y4-y3)**2)

            # DISTANCIA DE OJOS
                    x5 = int(face_landmarks.landmark[33].x * width)
                    y5 = int(face_landmarks.landmark[33].y * height)
                    x6 = int(face_landmarks.landmark[263].x * width)
                    y6 = int(face_landmarks.landmark[263].y * height)
                    d3=math.sqrt((x6-x5)**2+(y6-y5)**2)
                  
            # LARGO DE LA NARIZ
                    x7 = int(face_landmarks.landmark[8].x * width)
                    y7 = int(face_landmarks.landmark[8].y * height)
                    x8 = int(face_landmarks.landmark[4].x * width)
                    y8 = int(face_landmarks.landmark[4].y * height)
                    d4=math.sqrt((x8-x7)**2+(y8-y7)**2)
                  
             # LARGO DE LOS LABIOS
                    x9 = int(face_landmarks.landmark[61].x * width)
                    y9 = int(face_landmarks.landmark[61].y * height)
                    x10 = int(face_landmarks.landmark[291].x * width)
                    y10 = int(face_landmarks.landmark[291].y * height)
                    d5=math.sqrt((x10-x9)**2+(y10-y9)**2)
                 
             # ANCHO DE CARA   
                    x11 = int(face_landmarks.landmark[93].x * width)
                    y11 = int(face_landmarks.landmark[93].y * height)
                    x12 = int(face_landmarks.landmark[323].x * width)
                    y12 = int(face_landmarks.landmark[323].y * height)
                    d6=math.sqrt((x12-x11)**2+(y12-y11)**2)
                
            # LARGO DE NARIZ
                    x13 = int(face_landmarks.landmark[102].x * width)
                    y13 = int(face_landmarks.landmark[102].y * height)
                    x14 = int(face_landmarks.landmark[331].x * width)
                    y14 = int(face_landmarks.landmark[331].y * height)
                    d7=math.sqrt((x14-x13)**2+(y14-y13)**2)
                   
            # ANCHO DE NARIZ
                    x15 = int(face_landmarks.landmark[0].x * width)
                    y15 = int(face_landmarks.landmark[0].y * height)
                    x16 = int(face_landmarks.landmark[17].x * width)
                    y16 = int(face_landmarks.landmark[17].y * height)
                    d8=math.sqrt((x16-x15)**2+(y16-y15)**2)
                  
            # ANCHO DE OJOS
                    x17 = int(face_landmarks.landmark[159].x * width)
                    y17 = int(face_landmarks.landmark[159].y * height)
                    x18 = int(face_landmarks.landmark[145].x * width)
                    y18 = int(face_landmarks.landmark[145].y * height)
                    d9=math.sqrt((x18-x17)**2+(y18-y17)**2)
                    
            # LARGO DE CARA
                    x19 = int(face_landmarks.landmark[10].x * width)
                    y19 = int(face_landmarks.landmark[10].y * height)
                    x20 = int(face_landmarks.landmark[152].x * width)
                    y20 = int(face_landmarks.landmark[152].y * height)
                    d10=math.sqrt((x20-x19)**2+(y20-y19)**2)
                  
            # DISTANCIA DEL LABIO AL MENTON
                    x21 = int(face_landmarks.landmark[17].x * width)
                    y21 = int(face_landmarks.landmark[17].y * height)
                    x22 = int(face_landmarks.landmark[152].x * width)
                    y22 = int(face_landmarks.landmark[152].y * height)
                    d11=math.sqrt((x22-x21)**2+(y22-y21)**2)

                    x23 = int(face_landmarks.landmark[244].x * width)
                    y23 = int(face_landmarks.landmark[244].y * height)
                    x24 = int(face_landmarks.landmark[464].x * width)
                    y24 = int(face_landmarks.landmark[464].y * height)
                    d12=math.sqrt((x24-x23)**2+(y24-y23)**2)

                    x25 = int(face_landmarks.landmark[143].x * width)
                    y25 = int(face_landmarks.landmark[143].y * height)
                    x26 = int(face_landmarks.landmark[372].x * width)
                    y26 = int(face_landmarks.landmark[372].y * height)
                    d13=math.sqrt((x26-x25)**2+(y26-y25)**2)

                    x27 = int(face_landmarks.landmark[70].x * width)
                    y27 = int(face_landmarks.landmark[70].y * height)
                    x28 = int(face_landmarks.landmark[300].x * width)
                    y28 = int(face_landmarks.landmark[300].y * height)
                    d14=math.sqrt((x28-x27)**2+(y28-y27)**2)

                    x29 = int(face_landmarks.landmark[57].x * width)
                    y29 = int(face_landmarks.landmark[57].y * height)
                    x30 = int(face_landmarks.landmark[287].x * width)
                    y30 = int(face_landmarks.landmark[287].y * height)
                    d15=math.sqrt((x30-x29)**2+(y30-y29)**2)
                    
                    x31 = int(face_landmarks.landmark[57].x * width)
                    y31 = int(face_landmarks.landmark[57].y * height)
                    x32 = int(face_landmarks.landmark[287].x * width)
                    y32 = int(face_landmarks.landmark[287].y * height)
                    d15=math.sqrt((x32-x31)**2+(y32-y31)**2)


                    """print("Distance1: ", d1)
                    print("Distance2: ", d2)
                    print("Distance3: ", d3)
                    print("Distance4: ", d4)
                    print("Distance5: ", d5)
                    print("Distance6: ", d6)
                    print("Distance7: ", d7)
                    print("Distance8: ", d8)
                    print("Distance9: ", d9)
                    print("Distance10: ", d10)
                    print("Distance11: ", d11)
                    print("Distance12: ", d12)
                    print("Distance13: ", d13)
                    print("Distance14: ", d14)
                    print("Distance14: ", d15)
                    print("Distance14: ", d16)
                    print("Distance14: ", d17)
                    print("Distance14: ", d18)
                    print("Distance14: ", d19)
                    print("Distance14: ", d20)"""
                    
       
                    md1max=41
                    md1min=30
                    md2max=3.16
                    md2min=2.23
                    md3max=137.13
                    md3min=101
                    md4max=64.19
                    md4min=43.41
                    md5max=74
                    md5min=50
                    md6max=194.25
                    md6min=154.11
                    md7max=68
                    md7min=45
                    md8max=28
                    md8min=17
                    md9max=12.36
                    md9min=8
                    md10max=218.22
                    md10min=168.35
                    md11max=38
                    md11min=37
                    md12max=36
                    md12min=24
                    md13max=35
                    md13min=23
                    md14max=172
                    md14min=125
                    md15max=96
                    md15min=67

                    ref1=0
                    ref2=0
                    ref3=0
                    ref4=0
                    ref5=0
                    ref6=0
                    ref7=0
                    ref8=0
                    ref9=0
                    ref10=0
                    ref11=0
                    ref12=0
                    ref13=0
                    ref14=0
                    ref15=0
                    ref16=0

                    if d1<=md1max:
                        if d1>=md1min:
                            ref1=1
                    else:
                        ref1=0

                    if d2<=md2max:
                        if d2>=md2min:
                            ref2=1
                    else:
                        ref2=0

                    if d3<=md3max:
                        if d3>=md3min:
                            ref3=1
                    else:
                        ref3=0

                    if d4<=md4max:
                        if d4>=md4min:
                            ref4=1
                    else:
                        ref4=0

                    if d5<=md5max:
                         if d5>=md5min:
                            ref5=1
                    else:
                        ref5=0

                    if d6<=md6max:
                        if d6>=md6min:
                            ref6=1
                    else:
                        ref6=0
    
                    if d7<=md7max:
                        if d7>=md7min:
                            ref7=1
                    else:
                        ref7=0

                    if d8<=md8max:
                        if d8>=md8min:
                            ref8=1
                    else:
                        ref8=0

                    if d9<=md9max:
                        if d9>=md9min:
                            ref9=1
                    else:
                        ref9=0

                    if d10<=md10max:
                        if d10>=md10min:
                            ref10=1
                    else:
                        ref10=0

                    if d11<=md11max:
                        if d11>=md11min:
                           ref11=1
                    else:
                        ref11=0

                    if d12<=md12max:
                        if d12>=md12min:
                           ref12=1
                    else:
                        ref12=0

                    if d13<=md13max:
                        if d13>=md13min:
                           ref13=1
                    else:
                        ref13=0

                    if d14<=md14max:
                        if d14>=md14min:
                           ref14=1
                    else:
                        ref14=0

                    if d15<=md15max:
                        if d15>=md15min:
                           ref15=1
                    else:
                        ref15=0

                    sumaTotal=((ref1+ref2+ref3+ref4+ref5+ref6+ref7+ref8+ref9+ref10+ref11+ref12+ref13+ref14+ref15)*100)/15
                    print("Porcentage: ", sumaTotal)    

            
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

cap.release()
cv2.destroyAllWindows()
                 