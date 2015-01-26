import camera, face_detector
import debug_render
import transport

def main():
   for frame, grey_frame in camera.get_frames():
      face_rects, face_points = face_detector.detect_faces(grey_frame)

      transport.send_json(face_points)
      
      debug_render.draw_features(frame, {
         'face_rects': face_rects
      })

if __name__ == '__main__':
   main()
