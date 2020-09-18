#import sys
#sys.path.append('c:/users/kaustubh/appdata/local/programs/python/python37-32/lib/site-packages')
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = './selfie_data/'
skip=0
file_name = input("enter the name of person: ")
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #1.3=scaling factor and 5 = no of neighbours
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    #print(faces)
    #cv2.imshow("Gray Frame",gray_frame)

    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        #Extraction (Crop out the required face) : Region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    
    cv2.imshow("Video Frame",frame)
    #cv2.imshow("Face section",face_section)
	#wait for the user input - q, then you'll stop the loop
	
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#converting our face list into a numpy array
face_data=np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
#saving the captured selfie intp the system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully save at"+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()