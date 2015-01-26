import cv2, numpy as np

FACE_CASCADE = cv2.CascadeClassifier('face_frontal.xml')

def detect_cascade(img, cascade, flags=None):
   if flags:
      flags = flags | cv2.CASCADE_SCALE_IMAGE
   else:
      flags = cv2.CASCADE_SCALE_IMAGE

   rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=flags)
   if len(rects) == 0:
      return []
   rects[:,2:] += rects[:,:2]
   return rects

def detect_faces(frame):
   # Capture frame-by-frame
   frame_h, frame_w = frame.shape

   # detect frontal faces
   face_rects = detect_cascade(frame, FACE_CASCADE, cv2.CASCADE_DO_CANNY_PRUNING)
   face_points = [[float((x1+x2)/2.),float((y1+y2)/2.),float((np.abs(y2-y1)))] for x1, y1, x2, y2 in face_rects]

   return face_rects, face_points
