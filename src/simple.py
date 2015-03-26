import camera
import debug_render
import time
import math
import transport
import face_detector
import pose_inference
import template_matching
import correspondence

MAX_MATCHES = 50

def main():
   debug_render.init()

   faces = []
   face_data = []
   last_frame = None

   for frame in camera.get_frames(source=0, props=None):
      grey_frame = camera.greyscale(frame)
      new_faces = face_detector.detect_faces(grey_frame)
      
      if len(new_faces) > 0:
         # Calculate corresponding features in adjacent frames
         corresponding_faces = correspondence.correspond(face_data, faces, new_faces)
         missing_faces_data = corresponding_faces['missing_features']
         faces = corresponding_faces['features']
         face_data = corresponding_faces['feature_data']
      else:
         missing_faces_data = face_data
         faces = []
         face_data = []

      # Reset properties for detected faces
      for face_data_point in face_data:
         face_data_point.update({
            'mode': 'detected',
            'last_detected_as': face_data_point['feature'],
            'matches_made': 0,
            'alive_for': face_data_point.get('alive_for', 0) + 1
         })

      # Set properties for inferred faces
      for missing_face_data_point in missing_faces_data:
         missing_face_data_point['mode'] = 'inferred'
         if 'last_detected_as' not in missing_face_data_point:
            missing_face_data_point['last_detected_as'] = missing_face_data_point['feature']
         if 'matches_made' in missing_face_data_point:
            missing_face_data_point['matches_made'] += 1
         else:
            missing_face_data_point['matches_made'] = 1

      # Filter out old inferred faces that are really old
      missing_faces_data = [data for data in missing_faces_data if data['matches_made'] <= MAX_MATCHES]
      missing_faces = [data['feature'] for data in missing_faces_data]

      # Infer where missing faces have moved using template matching
      inferred_faces = []
      inferred_face_data = []
      flow = None
      if last_frame is not None and len(missing_faces) > 0:
         inferred_features = template_matching.template_match_features(last_frame, grey_frame, missing_faces, missing_faces_data)

         if len(inferred_features) > 0:
            inferred_faces, inferred_face_data = zip(*inferred_features)
            faces += inferred_faces
            face_data += inferred_face_data

      if last_frame is not None:
         flow = pose_inference.calc_flow(last_frame, grey_frame)

      if flow is not None:
         debug_render.draw_flow(frame, flow)

      debug_render.faces(frame, face_data)
      debug_render.draw_frame(frame)

      last_frame = grey_frame
      

if __name__ == '__main__':
   main()
