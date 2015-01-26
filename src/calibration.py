import camera
import debug_render
import cv2

def calibrate():
   debug_render.init()
   
   for frame, grey_frame in camera.get_frames(capture_key='c'):
      ret, corners = cv2.findChessboardCorners(grey_frame, (7,7))
      
      features = {}
      
      if ret:
         features['chessboard'] = corners

      debug_render.draw_features(frame, features)

if __name__ == '__main__':
   calibrate()
