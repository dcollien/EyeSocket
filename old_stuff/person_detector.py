import cv2, numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame):
   people_found, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

   people = zip(people_found, weights)
   return people
