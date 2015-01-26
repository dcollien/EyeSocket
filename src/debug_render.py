import cv2


def init():
   cv2.namedWindow('Video', flags=cv2.WINDOW_OPENGL)
   cv2.resizeWindow('Video', 640, 480)

def draw_features(frame, features):
   if 'face_rects' in features:
      face_rects = features['face_rects']
      
      color = (255, 0, 0)
      
      for x1, y1, x2, y2 in face_rects:
         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
   elif 'chessboard' in features:
      cv2.drawChessboardCorners(frame, (7,7), features['chessboard'], True)

   cv2.imshow('Video', frame)
