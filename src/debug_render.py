import cv2


def init():
   cv2.namedWindow('Video', flags=cv2.WINDOW_OPENGL)
   cv2.resizeWindow('Video', 640, 480)

def draw_frame(frame):
   cv2.imshow('Video', frame)

def draw_features(frame, features):
   if 'face_rects' in features:
      face_rects = features['face_rects']
      
      color = (255, 0, 0)
      
      for x1, y1, x2, y2 in face_rects:
         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
   
   if 'predictions' in features:
      for uuid, vect, confidence in features['predictions']:
         (x, y, size) = vect[:3]
         
         variance = (confidence[0] + confidence[1] / 2.0)

         color = (0, 255, 0)
         if variance > 0.5:
            color = (128, 128, 0)
         
         cv2.circle(frame, (int(x), int(y)), int(size/2.), color, 3)   
         cv2.putText(frame, str(uuid), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

   if 'face_points' in features:
      face_points = features['face_points']

      color = (255, 0, 0)
      for x, y, size in face_points:
         cv2.circle(frame, (int(x), int(y)), int(size/2.), color, 2)

   if 'guessed_face_points' in features:
      face_points = features['guessed_face_points']

      color = (0, 0, 255)
      for x, y, size in face_points:
         cv2.circle(frame, (int(x), int(y)), int(size/2.), color, 2)

   if 'chessboard' in features:
      cv2.drawChessboardCorners(frame, features['chessboard']['dimensions'], features['chessboard']['corners'], True)

   draw_frame(frame)

def wait_for_key(key='q'):
   while chr(cv2.waitKey(1) & 0xFF) != key:
      pass
