import cv2, numpy as np

from scipy.spatial import cKDTree as KDTree

def template_match_features(frame1, frame2, frame_1_features, frame_1_data=None, scale=1.5):
	new_positions = []

	w, h = frame1.shape

	# Constrains an x,y value into the width/height of the current frame
	def constrain(x_val, y_val):
		return min(w-1, max(0, x_val)), min(h-1, max(0, y_val))
	
	for i, feature in enumerate(frame_1_features):
		(x, y, size) = feature

		try:
			# constrain the feature coordinates to be inside the image
			x1, y1 = constrain(x - size/2, y - size/2)
			x2, y2 = constrain(x + size/2, y + size/2)

			# increase the size of the feature to a larger search area
			# surrounding the feature
			search_x1, search_y1 = constrain(x - size * scale, y - size * scale)
			search_x2, search_y2 = constrain(x + size * scale, y + size * scale)

			# the feature and search area regions of interest
			feature_roi  = frame1[y1:y2, x1:x2]
			search_roi = frame2[search_y1:search_y2, search_x1:search_x2]

			print (y1, y2, x1, x2)
			print (search_y1, search_y2, search_x1, search_x2)
			
			# match the feature image in the search area, and gets its x,y position
			result = cv2.matchTemplate(search_roi, feature_roi, cv2.TM_SQDIFF_NORMED)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
			(x, y) = (min_loc[0] + search_x1 + size/2, min_loc[1] + search_y1 + size/2)

			if frame_1_data is None:
				new_positions.append((x, y, size))
			else:
				frame_1_data[i]['feature'] = (x, y, size)
				new_positions.append(((x, y, size), frame_1_data[i]))
		except Exception:
			pass

	return new_positions
