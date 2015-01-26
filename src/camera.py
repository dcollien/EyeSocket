import cv2

def get_frames():
	video_capture = cv2.VideoCapture(0)

	while True:
		ret, frame = video_capture.read()

		grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grey_frame = cv2.equalizeHist(grey_frame)

		yield (frame, grey_frame)

	video_capture.release()
