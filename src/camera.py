import cv2

DEFAULT_CAP_PROPS = {
   cv2.CAP_PROP_FRAME_WIDTH: 640,
   cv2.CAP_PROP_FRAME_HEIGHT: 400,
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

def get_frames(quit_key='q', source=0, props=DEFAULT_CAP_PROPS):
   video_capture = cv2.VideoCapture(source)
   
   if props:
      for prop in props:
         video_capture.set(prop, props[prop])

   while True:
      key = cv2.waitKey(1) & 0xFF

      ret, frame = video_capture.read()

      if key == ord(quit_key):
         # wait for quit key to be pressed
         break
      
      yield frame

   video_capture.release()
