import cv2
import numpy as np
import math

TAU = math.pi * 2

def init():
   cv2.namedWindow('Video', flags=cv2.WINDOW_OPENGL)
   cv2.resizeWindow('Video', 640, 480)

def draw_frame(frame):
   cv2.imshow('Video', frame)

def faces(frame, faces):
   for face in faces:
      mode = face.get('mode')
      if mode == 'inferred':
         color = (0, 0, 255)
      else:
         color = (255, 0, 0)

      if face.get('alive_for', 0) < 5:
         color = (0, 0, 0)

      x, y, size = face['feature']
      cv2.circle(frame, (int(x), int(y)), int(size/2.), color, 2)
      cv2.putText(frame, str(face['id']), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
      cv2.putText(frame, str(face['alive_for']), (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
      
      cv2.putText(frame, str(face['matches_made']), (int(x), int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def draw_action_regions(frame, regions):
   h, w = frame.shape[:2]

   for region in regions:
      x1, x2, face = region
      x1, x2 = int(x1), int(x2)

      fx, fy, fs = face['feature']
      fy = int(fy)

      cv2.line(frame, (x1, 0), (x1, h), (255, 255, 0), 3)
      cv2.line(frame, (x2, 0), (x2, h), (0, 255, 255), 3)
      cv2.line(frame, (x1, fy), (x2, fy), (255, 0, 255), 2)


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

def draw_flow(img, flow, step=8):
   h, w = img.shape[:2]
   y, x = np.mgrid[step/2:h/2:step, step/2:w/2:step].reshape(2,-1)
   x = x.astype(int)
   y = y.astype(int)
   fx, fy = flow[y,x].T
   lines = np.vstack([x*2, y*2, (x+fx)*2, (y+fy)*2]).T.reshape(-1, 2, 2)
   lines = np.int32(lines + 0.5)

   # thresholding of the flow
   threshold = 8
   threshold = threshold**2
   lines = [np.array([(x1, y1), (x2, y2)]) for (x1, y1), (x2, y2) in lines if (x2 - x1)**2 + (y2 - y1)**2 > threshold]

   cv2.polylines(img, lines, 0, (0, 255, 0))
   for (x1, y1), (x2, y2) in lines:
      color = (0, 255, 0)
      cv2.circle(img, (x1, y1), 2, color, -1)

def wait_for_key(key='q'):
   while chr(cv2.waitKey(1) & 0xFF) != key:
      pass




# Old stuff

"""
def people(frame, people):
   color = (0, 255, 0)
   for person in people:
      rect, weight = person
      (x1, y1, x2, y2) = rect
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cv2.putText(frame, str(weight), (int(x1), int(y1 + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
"""

"""
def draw_pose(img, pose, dimensions_delegate):
   pose['l_forearm'] = TAU/4
   pose['r_forearm'] = TAU/4
   


   head_x, head_y, head_h = pose['head']

   head_x, head_y, head_h = (int(head_x), int(head_y), int(head_h))

   dimensions = dimensions_delegate(pose['head'])

   neck_length = dimensions['neck']
   shoulder_length = dimensions['shoulder']
   forearm_length  = dimensions['forearm']


   cv2.circle(img, (head_x, head_y), int(head_h/2.), (0, 255, 0), 2)

   chin  = (head_x, int(head_y + head_h/2.))
   chest = (chin[0], chin[1] + neck_length)
   l_shoulder = (chest[0] - shoulder_length, chest[1])
   r_shoulder = (chest[0] + shoulder_length, chest[1])

   vect_l_elbow = (math.cos(-pose['l_forearm']) * forearm_length, math.sin(-pose['l_forearm']) * forearm_length)
   l_elbow = (int(l_shoulder[0] - vect_l_elbow[0]), int(l_shoulder[1] - vect_l_elbow[1]))

   vect_r_elbow = (math.cos(pose['r_forearm']) * forearm_length, math.sin(pose['r_forearm']) * forearm_length)
   r_elbow = (int(r_shoulder[0] + vect_r_elbow[0]), int(r_shoulder[1] + vect_r_elbow[1]))

   left_color   = (255, 0, 0)
   left_color2  = (128, 0, 0)
   right_color  = (0, 0, 255)
   right_color2 = (0, 0, 128)

   cv2.line(img, chin, chest, (0, 255, 0), 2)
   cv2.line(img, chest, l_shoulder, left_color2, 2)
   cv2.line(img, chest, r_shoulder, right_color2, 2)
   cv2.line(img, l_shoulder, l_elbow, left_color, 2)
   cv2.line(img, r_shoulder, r_elbow, right_color, 2)
"""
