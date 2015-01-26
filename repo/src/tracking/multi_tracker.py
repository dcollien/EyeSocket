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
	# strikes:           how many times a predictor is flagged as to be discarded (in a row),
	#                    before it is deleted
	def __init__(self, remove_threshold=200, add_threshold=50, d=2, strikes=10):
		self.num_tracking = 0
		self.filters = []
		self.remove_threshold = remove_threshold
		self.add_threshold = add_threshold
		self.last_id = 0
		self.d = d
		self.strikes = strikes
		self.filter_data = {}

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

		# Reset the number of "strikes" against this filter
		self.filter_data[predictor_filter] = {
			"strikes": self.strikes,
		}

	def strike_filter(self, predictor_filter):
		# make a strike against this filter as not being accurate
		filter_data = self.filter_data[predictor_filter]
		filter_data['strikes'] -= 1
		return filter_data['strikes']

	def unstrike_filter(self, predictor_filter):
		# this filter is good again
		self.filter_data[predictor_filter]['strikes'] = self.strikes

	def is_new_observation(self, observation):
		# ignore observations that have been assigned/recorded
		if tuple(observation) in self.recorded:
			return False

		# find the closest predictor (filter) to the orphaned observation
		closest_prediction_distance = np.min(cdist([observation], self.predictions)[0])
		return closest_prediction_distance > self.add_threshold

	def predict(self):
		# tell each filter to make a prediction based on current data
		self.predictions = np.array([predictor.predict()[:self.d] for predictor in self.filters])
		return self.predictions

	def observe(self, observations):
		# update each filter with its closest observation
		# uses a KDTree to calculate closest points

		if self.num_tracking == 0:
			# not tracking anything yet, make a bunch of trackers (filters) for each observation
			self.filters = [self.make_filter(observation) for observation in observations]

		self.num_tracking = len(self.filters)

		# Pack all the observation coordinates into a KDTree
		observations = KDTree(observations)

		# nearest is a list of the index of the closest observation for each prediction
		# distances is a list of the distance from each observation to each prediction respectively
		distances, nearest = observations.query(self.predictions, distance_upper_bound=self.remove_threshold)

		predicted_observations = {}
		dead = set()
		self.recorded = set()
		size = observations.data.size
		for pi, oi in enumerate(nearest):
			# pi is the index of a prediction in "nearest"
			# oi is the index of the closest observation to that prediction in observations.data

			if distances[pi] < self.remove_threshold and oi < size:
				# this distance is close enough that we want to process it

				# get the last prediction distance, and the last matched prediction index
				# for this observation, defaults to (infinity, -1) respectively
				last_prediction_dist, last_pi = predicted_observations.get(oi, (np.inf, -1))
				if distances[pi] < last_prediction_dist:
					# this prediction is closer than another matched to this observation
					# delete the other
					if last_pi >= 0:
						dead.add(last_pi)

					# mix this observation into the filter of this predictor
					observation = observations.data[oi]
					self.filters[pi].observe(observation)

					# ensure this filter is all good
					self.unstrike_filter(self.filters[pi])

					# mark this observation as having been recorded by a filter
					self.recorded.add(tuple(observation))

					# set this one as the closest
					predicted_observations[oi] = (distances[pi], pi)
				elif self.strike_filter(self.filters[pi]) <= 0:
					# this prediction is further away than another better one,
					# delete it
					dead.add(pi)

			elif self.strike_filter(self.filters[pi]) <= 0:
				# after a few strikes, this filter is deleted (too far away from anything)
				dead.add(pi)

		# filter out predictors (filters) marked as "dead"
		self.filters = [predictor for pi, predictor in enumerate(self.filters) if pi not in dead]

		# add any new, orphaned observations as new filters
		self.filters += [self.make_filter(obs) for obs in observations.data if self.is_new_observation(obs)]

		# return an identifier and the last position estimate for each prediction
		return [(predictor.id, predictor.last_estimate) for predictor in self.filters]
