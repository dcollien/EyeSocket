import camera
import debug_render
import time
import math
import transport
import face_detector
import action_detector
import template_matching
import correspondence
import sys

import cv2
import copy

import json


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

   vx, vy = feature.get('v', (0,0))

   action = feature.get('action', 'still')

   if action == 'energy_left':
      action = 'wave_left'
   elif action == 'energy_right':
      action = 'wave_right'

   is_interesting = 1 if action_detector.is_interesting(feature) else 0

   return (feature['id'], x, y, size, mode, action, faces_matched, guesses_made, vx, vy, is_interesting)

def filter_features(features, max_features=None):
   filtered_features = []

   sorted_features = sorted(features, key=lambda x: x.get('distance', 0))

   for feature in sorted_features:
      if feature['alive_for'] > 5 and feature.get('matches_made', 0) < 15:
         filtered_features.append(feature)

   if max_features is not None:
      filtered_features = filtered_features[:max_features]

   return filtered_features

def main(config):

   MAX_MATCHES = config['max_matches']
   USE_SCALING_ALTERNATION = config['use_scaling_alternation']
   show_debug = not config['headless']

   faces = []
   face_data = []
   last_frame = None

   cameras = camera.set_up_cameras(config['cameras'])

   focal_length = config['focal_length']

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
      face_size_ranges = [[(200, 200), (20, 20)]]
      face_scale = 1.2

   frame_index = 0

   cam_args = {}

   if show_debug:
      debug_render.init()

   while True:
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
         # wait for quit key to be pressed
         break

      frame = camera.get_blended_frame(cameras)
      grey_frame = camera.greyscale(frame)

      if show_debug:
         debug_frame = copy.copy(grey_frame)
         debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)

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
         curr_x, curr_y, curr_s = face_data_point['feature']

         if not face_data_point.get('has_moved', False):
            has_moved = False
            if 'last_detected_as' in face_data_point:
               last_x, last_y, last_s = face_data_point['last_detected_as']
               if ((last_x - curr_x)**2 + (last_y - curr_y)**2) > 10:
                  has_moved = True

            # see if this face has ever moved
            is_moving = False
            if 'v' in face_data_point:
               vx, vy = face_data_point['v']
               if (vx ** 2 + vy ** 2) > 3:
                  is_moving = True

            face_data_point['has_moved'] = (has_moved and is_moving)

         face_data_point.update({
            'mode': 'detected',
            'last_detected_as': face_data_point['feature'],
            'matches_made': 0,
            'alive_for': face_data_point.get('alive_for', 0) + 1,
            'distance': camera.get_distance(curr_s, focal_length)
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

      if show_debug:
         # render pretty face boxes onto the colored frame
         debug_render.faces(debug_frame, face_data)

      # detect movement actions from the optic flow and face positions
      action_regions  = action_detector.get_action_regions(face_data)

      if last_frame is not None:
         # calculate the optic flow of the frame, at a low sample rate
         flow = action_detector.calc_flow(last_frame, grey_frame)

      if flow is not None and show_debug:
         # render a pretty flow onto the colored frame
         debug_render.draw_flow(debug_frame, flow)

      # calculate face velocities
      for face in face_data:
         face['v'] = action_detector.get_face_velocity(frame, flow, face)

      if show_debug:
         debug_render.draw_action_regions(debug_frame, action_regions)

      action_detector.detect_actions(grey_frame, flow, action_regions)

      features_to_send = filter_features(face_data, config.get('max_features', None))

      packed_features = [pack_feature(feature, (frame_w, frame_h)) for feature in features_to_send]


      # keep track of the last frame (for flow and template matching)
      last_frame = grey_frame

      # send the features over the network
      transport.send_features(packed_features)

      if show_debug:
         debug_render.draw_actions(debug_frame, action_regions)

         # draw the modified color frame on the screen
         debug_render.draw_frame(debug_frame)

      frame_index = (frame_index + 1) % len(face_size_ranges)

   camera.release_cams(cameras)

if __name__ == '__main__':
   args = sys.argv[1:]

   if len(args) > 0:
      conf_file = args[0]
   else:
      conf_file = 'config.json'

   with open(conf_file) as config_data:
      config = json.loads(config_data.read())
      main(config)
