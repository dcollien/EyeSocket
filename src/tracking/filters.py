import cv2
import numpy as np
import uuid

# Simplistic filter for testing
class SimpleFilter(object):
   def __init__(self, observation, max_distance=500):
      self.id = uuid.uuid1()
      self.max_distance = max_distance
      self._confidence = np.ones_like(observation)

      observation = np.array(observation)
      self.last_estimate = observation
      self.last_prediction = observation
      self.estimate_last_prediction = observation

   def predict(self):
      velocity = self.last_estimate - self.estimate_last_prediction
      self.last_prediction = self.last_estimate + velocity
      self.estimate_last_prediction = self.last_estimate
      return self.last_prediction

   def observe(self, observation):
      observation = np.array(observation)
      self._confidence = np.ones_like(observation)# ((np.ones_like(observation) * self.max_distance) - (self.last_estimate - observation))/self.max_distance
      self.last_estimate = observation
      return observation

   def confidence(self):
      return self._confidence

   def __eq__(self, other):
      return self.id == other.id

   def __ne__(self, other):
      return not self.__eq__(other)

   def __hash__(self):
      return hash(self.id)

# Kalman filter base class
class Filter(object):
   def __init__(self, dynamic, measurement, control):
      self.id = uuid.uuid1()
      self.last_estimate   = None
      self.last_prediction = None
      self.last_observation = None

      self.kalman = cv2.KalmanFilter(dynamic, measurement, control)
      self.reset()

   def predict(self):
      self.last_prediction = self.kalman.predict()
      self.last_prediction = tuple(self.last_prediction[:,0])
      return self.last_prediction

   def reset(self):
      raise NotImplementedError

   def observe(self, observation):
      raise NotImplementedError

   def confidence(self):
      return self.kalman.errorCovPost[0]

   def __eq__(self, other):
      return self.id == other.id

   def __ne__(self, other):
      return not self.__eq__(other)

   def __hash__(self):
      return hash(self.id)

class Filter2D(Filter):
   """ Simple 2D Motion Filter """
   def __init__(self, observation=None):
      # Create a kalman filter with:
      # - 4 dynamic parameters
      # - 2 measurement parameters
      # - 0 control parameters
      super(Filter2D, self).__init__(4, 2, 0)

      if observation is not None:
         self.kalman.statePost = np.array([
            [observation[0]],
            [observation[1]],
            [0.],
            [0.]
         ])
         self.last_estimate = tuple(self.kalman.statePost[:,0])
   
   def reset(self):
      dps = 4
      mps = 2

      kalman = self.kalman
      kalman.transitionMatrix = np.array([
         [1., 0., 1., 0.],
         [0., 1., 0., 1.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]
      ])
      kalman.measurementMatrix = np.array([
         [1., 0., 1., 0.],
         [0., 1., 0., 1.]
      ])
      kalman.processNoiseCov = 1e-4 * np.eye(dps)
      kalman.measurementNoiseCov = 1e-1 * np.eye(mps)
      kalman.errorCovPost = .1 * np.ones((dps, dps))
      kalman.statePost = .1 * np.random.randn(dps, 1.)

   def observe(self, observation):
      self.last_observation = observation

      x, y = observation
      x, y = float(x), float(y)

      observation = np.array([[x],[y]])
      self.last_estimate = self.kalman.correct(observation)
      self.last_estimate = tuple(self.last_estimate[:,0])
      return self.last_estimate


class Filter3D(Filter):
   """ Simple 3D Motion Filter """
   def __init__(self, observation=None):
      # Create a kalman filter with:
      # - 6 dynamic parameters
      # - 3 measurement parameters
      # - 0 control parameters
      super(Filter3D, self).__init__(6, 3, 0)

      if observation is not None:
         self.kalman.statePost = np.array([
            [observation[0]],
            [observation[1]],
            [observation[2]],
            [0.],
            [0.],
            [0.]
         ])
         self.last_estimate = tuple(self.kalman.statePost[:,0])
   
   def reset(self):
      dps = 6
      mps = 3

      kalman = self.kalman
      kalman.transitionMatrix = np.array([
         [1., 0., 0., 1., 0., 0.],
         [0., 1., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 1.],
         [0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 1.]
      ])
      kalman.measurementMatrix = np.array([
         [1., 0., 0., 1., 0., 0.],
         [0., 1., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 1.]
      ])
      kalman.processNoiseCov = 1e-4 * np.eye(dps)
      kalman.measurementNoiseCov = 1e-1 * np.eye(mps)
      kalman.errorCovPost = .1 * np.ones((dps, dps))
      kalman.statePost = .1 * np.random.randn(dps, 1.)

   def observe(self, observation):
      self.last_observation = observation

      x, y, z = observation
      x, y, z = float(x), float(y), float(z)

      observation = np.array([[x],[y],[z]])
      self.last_estimate = self.kalman.correct(observation)
      self.last_estimate = tuple(self.last_estimate[:,0])
      return self.last_estimate
