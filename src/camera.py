import cv2

def get_frames(quit_key='q', capture_key=None):
   video_capture = cv2.VideoCapture(0)

   while True:
      key = cv2.waitKey(1) & 0xFF

      ret, frame = video_capture.read()

      grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      grey_frame = cv2.equalizeHist(grey_frame)

      if key == ord(quit_key):
         # wait for quit key to be pressed
         break
      elif capture_key is not None and key == ord(capture_key):
         # wait for capture key to be pressed
         yield (frame, grey_frame)
      elif capture_key is None:
         # just yield every frame
         yield (frame, grey_frame)

   video_capture.release()
