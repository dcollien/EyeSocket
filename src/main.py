import camera, face_detector, template_matching, pose_inference
import debug_render
import time
from tracking.multi_tracker import MultiTracker
import transport

PREDICTION_INTERVAL = 0.5

MAX_CONFIDENCE = 5

def main():
   debug_render.init()


   predictor_confidence = {}

   last_frame = None

   tracker = MultiTracker(remove_threshold=100, add_threshold=100, d=3)

   time_at_last_prediction = time.process_time()

   for frame in camera.get_frames(source=0):#source='../testing/motinas_multi_face_turning.avi'):
      grey_frame = camera.greyscale(frame)
      face_points = face_detector.detect_faces(grey_frame)

      #print(face_points)

      predictions = tracker.observe(face_points)

      # mark the latest predictions that were 
      # assigned observations as being confident
      for pi in tracker.assigned:
         predictor = tracker.lookup_filter(pi)
         #print('assigned', predictor.id)
         predictor_confidence[predictor] = MAX_CONFIDENCE

      # mark rubbish predictions as being less confident
      # and keep track of ones we want to remove
      marked_for_removal = set()
      for pi in tracker.rubbish:
         predictor = tracker.lookup_filter(pi)
         if predictor in predictor_confidence:
            # remove 1 confidence points
            predictor_confidence[predictor] -= 1

      # accumulate the points that were missing observations
      missing_faces = []
      for pi in tracker.missing:
         # find the last observation made by this predictor
         predictor = tracker.lookup_filter(pi)

         if predictor not in predictor_confidence:
            #print('added', predictor.id)
            predictor_confidence[predictor] = MAX_CONFIDENCE

         # remove 1 confidence point
         predictor_confidence[predictor] -= 1
         #print(predictor_confidence[predictor], predictor.id)

         if predictor is not None:
            last_observation = predictor.last_estimate[:3] # predictor.last_observation
            # determine how confident this predictor is
            confidence_vect = predictor.confidence()
            variance = (confidence_vect[0] + confidence_vect[1] / 2.0)
            if last_observation is not None and variance < 0.5:
               missing_faces.append(last_observation)

      # add new filters for ones that weren't assigned
      for observation in tracker.unassigned:
         tracker.add_filter(observation)

      flow = None
      inferred_faces = []
      if last_frame is not None:
         # try to infer from the previous frame where missing faces might have moved to
         inferred_faces = template_matching.template_match_features(last_frame, grey_frame, missing_faces)
         
         # add observations for each inferred face
         predictions = tracker.observe(inferred_faces)

         flow = pose_inference.calc_flow(last_frame, grey_frame)
      
      # remove filters that we have lost faith in
      tracker.remove_filters([predictor for predictor, confidence in predictor_confidence.items() if confidence <= 0])

      last_frame = grey_frame

      # make periodic predictions
      if time.process_time() - time_at_last_prediction > PREDICTION_INTERVAL:
         tracker.predict()
         time_at_last_prediction = time.process_time()

      if flow is not None:
         debug_render.draw_flow(frame, flow)

         if len(predictions) > 0:
            pred = predictions[0][1]
            head = (pred[0], pred[1], pred[2])
            #pose_inference.infer_pose(flow, head)
            #debug_render.draw_pose(frame, {'head': head}, pose_inference.get_dimensions)

      debug_render.draw_features(frame, {
         'face_points': face_points,
         'guessed_face_points': inferred_faces,
         'predictions': predictions
      })



      
      output = [
         "{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\n".format(uuid, vect[0], vect[1], vect[2], confidence[0], confidence[1], confidence[2]) 
         for uuid, vect, confidence in predictions
      ]

      transport.send_data(''.join(output))
      
      

      

if __name__ == '__main__':
   main()
