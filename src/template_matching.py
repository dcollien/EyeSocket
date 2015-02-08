import cv2, numpy as np

from scipy.spatial import cKDTree as KDTree

def template_match_features(frame1, frame2, frame_1_features, scale=1.5):
	new_positions = []

	w, h = frame1.shape

	for i, feature in enumerate(frame_1_features):
		(x, y, size) = feature

		try:
			x1, y1 = max(0, x - size/2), max(0, y - size/2)
			x2, y2 = min(w-1, x + size/2), min(h-1, y + size/2)
			search_x1, search_y1 = max(0, x - size * scale), max(0, y - size * scale)
			search_x2, search_y2 = min(w-1, x + size * scale), min(h-1, y + size * scale)

			frame_roi  = frame1[y1:y2, x1:x2]
			search_roi = frame2[search_y1:search_y2, search_x1:search_x2]

			result = cv2.matchTemplate(search_roi, frame_roi, cv2.TM_SQDIFF_NORMED)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
			(x, y) = (min_loc[0] + search_x1 + size/2, min_loc[1] + search_y1 + size/2)

			new_positions.append((x, y, size))
		except Exception:
			pass

	return new_positions
