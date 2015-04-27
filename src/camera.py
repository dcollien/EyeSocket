import cv2
from time import sleep

DEFAULT_CAP_PROPS = {
   cv2.CAP_PROP_FRAME_WIDTH: 1400,#1280,
   cv2.CAP_PROP_FRAME_HEIGHT: 1080,#960,
   cv2.CAP_PROP_FPS: 25
}

TESTING_CAP_PROPS = {
   cv2.CAP_PROP_FRAME_WIDTH: 1280/2,
   cv2.CAP_PROP_FRAME_HEIGHT: 960/2,
   cv2.CAP_PROP_FPS: 25
}

def capture_image(filename):
   return cv2.imread(filename)

def capture_on_key(capture_keys=None, all_frames=False, quit_key='q'):
   video_capture = cv2.VideoCapture(0)

   while True:
      key = chr(cv2.waitKey(1) & 0xFF)

      ret, frame = video_capture.read()

      if capture_keys is None or key in capture_keys:
         yield (key, frame)
      elif key == quit_key:
         break
      elif all_frames:
         yield (False, frame)

   video_capture.release()

def greyscale(img):
   grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   grey_img = cv2.equalizeHist(grey_img)
   return grey_img

def get_frames(quit_key='q', source=0, props=DEFAULT_CAP_PROPS, crop=None):
   video_capture = cv2.VideoCapture(source)
   
   if props:
      for prop in props:
         video_capture.set(prop, props[prop])

      w, h = (props[cv2.CAP_PROP_FRAME_WIDTH], props[cv2.CAP_PROP_FRAME_HEIGHT])
   else:
      w, h = 1280, 960


   while True:
      key = cv2.waitKey(1) & 0xFF

      ret, frame = video_capture.read()

      if crop is not None:
         x1, y1, x2, y2 = crop
         x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
         frame = frame[y1:y2, x1:x2]

      if key == ord(quit_key):
         # wait for quit key to be pressed
         break
      
      yield frame

   video_capture.release()
