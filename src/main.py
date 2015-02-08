import camera, face_detector, template_matching
import debug_render
import time
from tracking.multi_tracker import MultiTracker
#import transport

PREDICTION_INTERVAL = 0.5

MAX_CONFIDENCE = 30

def main():
   debug_render.init()


   predictor_confidence = {}

   last_frame = None

   tracker = MultiTracker(remove_threshold=100, add_threshold=60, d=3)

   time_at_last_prediction = time.process_time()

   for frame in camera.get_frames(source='../testing/motinas_multi_face_turning.avi'):
      grey_frame = camera.greyscale(frame)
      face_points = face_detector.detect_faces(grey_frame)

      predictions = tracker.observe(face_points)

      # mark the latest predictions that were 
      # assigned observations as being confident
      for pi in tracker.assigned:
         predictor_confidence[pi] = MAX_CONFIDENCE

      # mark rubbish predictions as being less confident
      # and keep track of ones we want to remove
      marked_for_removal = set()
      for pi in tracker.rubbish:
         if pi in predictor_confidence:
            # remove 1 confidence points
            predictor_confidence[pi] -= 1

      # accumulate the points that were missing observations
      missing_faces = []
      for pi in tracker.missing:
         if pi not in predictor_confidence:
            predictor_confidence[pi] = MAX_CONFIDENCE

         # remove 1 confidence point
         predictor_confidence[pi] -= 2

         # find the last observation made by this predictor
         predictor = tracker.lookup_filter(pi)
         last_observation = predictor.last_observation
         # determine how confident this predictor is
         confidence_vect = predictor.confidence()
         variance = (confidence_vect[0] + confidence_vect[1] / 2.0)
         if last_observation is not None and variance < 0.5:
            missing_faces.append(last_observation)

      # remove filters that we have lost faith in
      tracker.remove_filters([pi for pi, confidence in predictor_confidence.items() if confidence <= 0])

      # add new filters for ones that weren't assigned
      for observation in tracker.unassigned:
         tracker.add_filter(observation)

      inferred_faces = []
      if last_frame is not None:
         # try to infer from the previous frame where missing faces might have moved to
         inferred_faces = template_matching.template_match_features(last_frame, grey_frame, missing_faces)
         
         # add observations for each inferred face
         predictions = tracker.observe(inferred_faces)
      
      last_frame = grey_frame

      # make periodic predictions
      if time.process_time() - time_at_last_prediction > PREDICTION_INTERVAL:
         tracker.predict()
         tracker.predict()
         tracker.predict()
         time_at_last_prediction = time.process_time()

      debug_render.draw_features(frame, {
         'face_points': face_points,
         'guessed_face_points': inferred_faces,
         'predictions': predictions
      })

      

      

if __name__ == '__main__':
   main()
