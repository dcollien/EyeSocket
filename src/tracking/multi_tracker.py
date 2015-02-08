from tracking.filters import Filter2D, Filter3D

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import cKDTree as KDTree

class MultiTracker(object):
   # remove_threshold:  predictors must have an observation within this distance (pixels)
   #                    before being flagged to be discarded
   # add_threshold:     observations must be further away than this distance (pixels)
   #                    to be given their own new predictor
   # d:                 dimensionality (default 2D space), can also be 3D
   def __init__(self, remove_threshold=100, add_threshold=50, d=2):
      self.filters = []
      self.remove_threshold = remove_threshold
      self.add_threshold = add_threshold
      self.last_id = 0
      self.d = d

      self.missing = set()
      self.rubbish = set()
      self.recorded = set()
      self.assigned = set()
      self.unassigned = []

   def make_filter(self, observation):
      # Create a new filter for this observation

      predictor_filter = None
      if self.d == 3:
         # 3D filter
         predictor_filter = Filter3D(observation)
      elif self.d == 2:
         # 2D filter
         predictor_filter = Filter2D(observation)
      else:
         raise NotImplementedError

      predictor_filter.predict()

      return predictor_filter

   def add_filter(self, observation):
      self.filters.append(self.make_filter(observation))

   def is_new_observation(self, observation):
      # ignore observations that have been assigned/recorded
      if tuple(observation) in self.recorded:
         return False

      # find the closest predictor (filter) to the orphaned observation
      closest_prediction_distance = np.min(cdist([observation], self.predictions)[0])
      return closest_prediction_distance > self.add_threshold

   def lookup_filter(self, index):
      try:
         return self.filters[index]
      except IndexError:
         return None

   def remove_filters(self, filters):
      # filter out predictors (filters)
      self.filters = [predictor for pi, predictor in enumerate(self.filters) if predictor not in frozenset(filters)]

   def predict(self):
      # tell each filter to make a prediction based on current data
      self.predictions = np.array([predictor.predict()[:self.d] for predictor in self.filters])
      return self.predictions

   def observe(self, observations):
      # update each filter with its closest observation
      # uses a KDTree to calculate closest points

      self.missing = set()
      self.rubbish = set()
      self.recorded = set()
      self.assigned = set()

      if len(observations) == 0:
         predictions = []
         self.missing = set()
         for i, predictor in enumerate(self.filters):
            predictions.append((predictor.id, predictor.last_estimate, tuple(predictor.confidence())))
            self.missing.add(i)
         return predictions

      self.num_tracking = len(self.filters)

      if self.num_tracking == 0:
         # not tracking anything yet, make a bunch of trackers (filters) for each observation
         self.filters = [self.make_filter(observation) for observation in observations]
         self.predict()

      # Pack all the observation coordinates into a KDTree
      observations = KDTree(observations)

      # nearest is a list of the index of the closest observation for each prediction
      # distances is a list of the distance from each observation to each prediction respectively

      self.predictions = np.array([predictor.last_prediction[:self.d] for predictor in self.filters])
      distances, nearest = observations.query(self.predictions, distance_upper_bound=self.remove_threshold)

      predicted_observations = {}
      
      size = observations.data.size

      # observations assigned to each predictor
      assigned_observations = {}

      # assign observations to predictors
      for pi, oi in enumerate(nearest):
         # pi is the index of a prediction in "nearest"
         # oi is the index of the closest observation to that prediction in observations.data

         if distances[pi] < self.remove_threshold and oi < size:
            # this distance is close enough that we want to process it

            # get the last prediction distance, and the last matched prediction index
            # for this observation, defaults to (infinity, -1) respectively
            last_prediction_dist, last_pi = predicted_observations.get(oi, (np.inf, -1))
            if distances[pi] < last_prediction_dist:
               # this prediction is closer to this observation
               # than any previously assigned prediction
               # so delete the other, and mark it as rubbish
               if last_pi >= 0 and pi in assigned_observations:
                  del assigned_observations[pi]
                  self.rubbish.add(pi)

               # assign the observation to this filter
               assigned_observations[pi] = observations.data[oi]

               # remember this one is the closest
               predicted_observations[oi] = (distances[pi], pi)
            else:
               # this prediction is further away than another, better one,
               # mark it as rubbish
               self.rubbish.add(pi)

         elif pi < self.num_tracking:
            # this filter hasn't got a nearby observation
            # mark it as missing
            self.missing.add(pi)

      for pi, observation in assigned_observations.items():
         self.filters[pi].observe(observation)
         # mark this observation as having been recorded by a filter
         self.recorded.add(tuple(observation))
         self.assigned.add(pi)

      # collect any new, orphaned observations as new filters
      self.unassigned = [obs for obs in observations.data if self.is_new_observation(obs)]

      # return an identifier and the last position estimate for each prediction
      return [(predictor.id, predictor.last_estimate, tuple(predictor.confidence())) for predictor in self.filters]
