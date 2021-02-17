# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:39:03 2021

@author: TAmilvannan G
"""
import cv2
import face_recognition
import pickle
import numpy as np


f = open("ref_name.pkl","rb")
ref_dict = pickle.load(f)
f.close()
ref_dict['unknown']="unknown"


f=open("ref_embed.pkl","rb")
embed_dict=pickle.load(f)
f.close()


known_face_encodings=[]
known_face_names=[]

for ref_id , embed_list in embed_dict.items():
    for embed in embed_list:
        known_face_encodings += [embed]
        known_face_names += [ref_id]
        
Video_capture = cv2.VideoCapture(0)     
face_locations =[]
face_encodings = []
face_names = []
process_this_frame = True

while True:
    
    ret, frame = Video_capture.read()
    
    #resize freame of video
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    
    #convert the image frome bgr 
    
    rgb_small_frame = small_frame[:, :, ::-1]
    
    
    if process_this_frame:
        #find all the faces and face recogniton
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_names = []
        
        for face_encodings in face_encodings:
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
            name = "unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                names = known_face_names[best_match_index]
            face_names.append(names)
            
    process_this_frame = not process_this_frame
    

    
    for(top,right, bottom,left), name in zip(face_locations,face_names):
        
        top *=4
        right *=4
        bottom *=4
        left *=4
        
        cv2.rectangle(frame, (left,top),(right,bottom),(0,0,255),2)
    
        cv2.rectangle(frame, (left,bottom -35), (right , bottom),(0,0,255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, ref_dict[name],(left+ 6, bottom -6), font, 1.0,(237, 234, 33),1)

    cv2.imshow('video',frame)
    
    if cv2.waitKey(1)==ord('q'):
        Video_capture.release()
        break
            
            