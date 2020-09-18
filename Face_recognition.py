#Face classifier
#We can use multiple algorithms for this such as Logistic regression, KNN, SVM etc
#In this model we'll use KNN to classify the faces data.
#Steps:-
#1 Load the training data (numpy array of all faces)
        # x-values are stored in numpy arrays
        # y-values we need to assign as ID for each person
#2 Read a video stream using opencv.
#3 Extract faces out of it
#4 use knn to find the prediction of face 
#5 map the predicted id to the face of user
#6 display the predictions on the screen - bounding box and name
import cv2
import numpy as np
import os

#------------KNN CODE---------------------------------############
def distance(v1,v2):
    #eucledian
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist=[]
    
    for i in range(train.shape[0]):
        #get the vector and label
        ix=train[i,:-1]
        iy=train[i,-1]
        # compute the distance and get the top k
        d = distance(test,ix)
        dist.append([d,iy])
    #sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
        # retrieve only labels
    labels = np.array(dk)[:,-1]
        
        
    #Get the frequencies of each label
    output = np.unique(labels,return_counts=True)
    #Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
###################

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
dataset_path='./selfie_data/'

face_data = []
label = []

class_id = 0 #labels for the given file
names = {} # maping btw id-name

#Data preperation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print('loaded '+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)
        
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(label,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

#Testing part

while True:
    ret,frame = cap.read()
    
    if ret==False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces:
        x,y,w,h=face
        
        #Get the Region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        out=knn(train_set,face_section.flatten())
        
        #Display on the screen the name and a rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)        
        
    cv2.imshow("faces",frame)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()