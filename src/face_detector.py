import cv2, numpy as np

FACE_CASCADE = cv2.CascadeClassifier('face_frontal.xml')

def detect_cascade(img, cascade, scale_factor, max_size, min_size):
   flags = cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING

   rects = cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=4, flags=flags, minSize=min_size, maxSize=max_size)
   if len(rects) == 0:
      return []
   rects[:,2:] += rects[:,:2]
   return rects

def detect_faces(frame, scale_factor=1.2, max_size=(200, 200), min_size=(10, 10)):
   max_height = 50

   # Capture frame-by-frame
   frame_h, frame_w = frame.shape

   # detect frontal faces
   face_rects = detect_cascade(frame, FACE_CASCADE, scale_factor=scale_factor, max_size=max_size, min_size=min_size)
   face_points = [[float((x1+x2)/2.),float((y1+y2)/2.),float((np.abs(y2-y1)))] for x1, y1, x2, y2 in face_rects]

   face_points = [(x, y, h) for x, y, h in face_points if h > max_height]

   return face_points
