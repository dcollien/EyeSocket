import camera
import debug_render
import cv2
import numpy as np

def gen_3d_coords():
   square_size = 0.25

   rows = [
      2, 1, 0, -1, -2, -3
   ]

   cols = [
      -3, -2, -1, 0, 1, 2, 3 
   ]

   corners = []
   for y in rows:
      for x in cols:
         corners.append((x * square_size, y * square_size, 0))

   return np.array(corners, dtype=np.float32)


def calibrate_camera(img_pts, obj_pts, img_size):
   # generate pattern size
   camera_matrix = np.zeros((3,3))
   dist_coef = np.zeros(4)
   rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera([obj_pts], [img_pts], img_size, camera_matrix, dist_coef)
   return camera_matrix, dist_coefs

def calibrate_3d(frame, features):
   ret, corners = cv2.findChessboardCorners(frame, (7,6))
   corners_3d = gen_3d_coords()

   if ret:
      features['chessboard'] = {
         'dimensions': (7,6),
         'corners': corners
      }


      corners = corners.reshape(7*6, 2).astype('float32')
      frame_h, frame_w = frame.shape  

      cam_matrix, dist_coefs = calibrate_camera(corners, corners_3d, (frame_w, frame_h))
      print(cv2.calibrationMatrixValues(cam_matrix, (frame_w, frame_h), 10.0, 10.0))
   else:
      print(ret)

def calibrate_from_image(filename):
   frame = camera.capture_image(filename)

   grey_frame = camera.greyscale(frame)
   
   features = {}
   
   calibrate_3d(grey_frame, features)


   debug_render.init()

   debug_render.draw_features(frame, features)

   debug_render.wait_for_key()

def calibrate():
   debug_render.init()

   is_rendering = True
   
   for key, frame in camera.capture_on_key(capture_keys=['c', 'v', 'z'], all_frames=True):

      if key in ['c', 'v']:
         grey_frame = camera.greyscale(frame)
         
         features = {}

         calibrate_3d(grey_frame, features)

         debug_render.draw_features(frame, features)

         is_rendering = False
      elif key == 'z':
         is_rendering = True
      elif is_rendering:
         debug_render.draw_frame(frame)

if __name__ == '__main__':
   calibrate_from_image('test_images/checkerboard_ground.jpg')
