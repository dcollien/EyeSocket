import camera, face_detector
import debug_render

def main():
   for frame, grey_frame in camera.get_frames():
      face_rects, face_points = face_detector.detect_faces(grey_frame)

      debug_render.draw_features(frame, {
         'face_rects': face_rects
      })

if __name__ == '__main__':
   main()
