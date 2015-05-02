import camera
import debug_render
import time
import math
import transport
import face_detector
import action_detector
import template_matching
import correspondence

MAX_MATCHES = 50
USE_SCALING_ALTERNATION = True # try and find faces of various sizes in alternate frames (speedup)

def pack_feature(feature, dimensions):
   x, y, size = feature['feature']

   w, h = dimensions

   x /= w
   y /= h

   y = (1.0 - y)

   mode = 0
   if feature.get('mode') == 'inferred':
      mode = 1

   guesses_made = feature['matches_made']
   faces_matched = feature['alive_for']
   
   action = feature.get('action', 'still')
   
   return (feature['id'], x, y, size, mode, action, faces_matched, guesses_made)

def main():
   debug_render.init()

   cropping = (0, 0.3, 1.0, 0.6)

   faces = []
   face_data = []
   last_frame = None

   if USE_SCALING_ALTERNATION:
      # generate which face sizes to look for in each frame
      face_size_ranges = []
      face_drop = 50
      max_size = 250
      min_size = 25
      face_scale = (max_size/float(max_size-face_drop/2))
      while max_size > min_size:
         face_size_ranges.append([(max_size, max_size), (max_size-face_drop, max_size-face_drop)])
         max_size -= face_drop
   else:
      face_size_ranges = [[(200, 200), (10, 10)]]
      face_scale = 1.2

   frame_index = 0

   #for frame in camera.get_frames(source=1, crop=cropping):#, props=camera.TESTING_CAP_PROPS):
   for frame in camera.get_frames(source=0, props=camera.TESTING_CAP_PROPS):
      grey_frame = camera.greyscale(frame)

      max_face_size, min_face_size = face_size_ranges[frame_index]

      new_faces = face_detector.detect_faces(grey_frame, scale_factor=face_scale, max_size=max_face_size, min_size=min_face_size)

      frame_h, frame_w = grey_frame.shape[:2]
      
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
         # there's some faces missing since the last frame, let's find where they went
         inferred_features = template_matching.template_match_features(last_frame, grey_frame, missing_faces, missing_faces_data)

         # we've inferred where some of them went, let's update our face data
         if len(inferred_features) > 0:
            inferred_faces, inferred_face_data = zip(*inferred_features)
            faces += inferred_faces
            face_data += inferred_face_data

      if last_frame is not None:
         # calculate the optic flow of the frame, at a low sample rate
         flow = action_detector.calc_flow(last_frame, grey_frame)

      if flow is not None:
         # render a pretty flow onto the colored frame
         debug_render.draw_flow(frame, flow)

      # render pretty face boxes onto the colored frame
      debug_render.faces(frame, face_data)

      # keep track of the last frame (for flow and template matching)
      last_frame = grey_frame

      # detect movement actions from the optic flow and face positions
      action_regions  = action_detector.get_action_regions(face_data)

      debug_render.draw_action_regions(frame, action_regions)
      
      action_detector.detect_actions(grey_frame, flow, action_regions)

      packed_features = [pack_feature(feature, (frame_w, frame_h)) for feature in face_data]

      # send the features over the network
      transport.send_features(packed_features)

      debug_render.draw_actions(frame, action_regions)

      # draw the modified color frame on the screen
      debug_render.draw_frame(frame)

      frame_index = (frame_index + 1) % len(face_size_ranges)

if __name__ == '__main__':
   main()
