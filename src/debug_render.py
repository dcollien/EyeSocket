import cv2

def draw_features(frame, features):
   face_rects = features['face_rects']
   
   color = (255, 0, 0)
   
   for x1, y1, x2, y2 in face_rects:
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

   cv2.imshow('Video', frame)
