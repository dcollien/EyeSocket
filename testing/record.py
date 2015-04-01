import cv2
import signal

width = 1280
height = 720

PROPS = {
   cv2.CAP_PROP_FRAME_WIDTH: width,
   cv2.CAP_PROP_FRAME_HEIGHT: height,
   cv2.CAP_PROP_FPS: 25
}

running = True
def signal_handler(signal, frame):
	global running
	running = False
signal.signal(signal.SIGINT, signal_handler)

capture_a = cv2.VideoCapture(1)
capture_b = cv2.VideoCapture(2)


for prop in PROPS:
	capture_a.set(prop, PROPS[prop])
	capture_b.set(prop, PROPS[prop])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer_a = cv2.VideoWriter("outputA.mov", fourcc, 25, (width, height), True)
video_writer_b = cv2.VideoWriter("outputB.mov", fourcc, 25, (width, height), True)

frames = 0
print('Recording...')
while (capture_a.isOpened() and capture_b.isOpened()) and running:
	ret_a, frame_a = capture_a.read()
	ret_b, frame_b = capture_b.read()
	if ret_a and ret_b:
		video_writer_a.write(frame_a)
		video_writer_b.write(frame_b)
		frames += 1


print("Recorded ", frames, 'frames')

capture_a.release()
capture_b.release()

video_writer_a.release()
video_writer_b.release()

print("Exiting...")
