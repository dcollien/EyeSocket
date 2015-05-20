import cv2, numpy as np
import math
from collections import deque

TAU = math.pi * 2
PI = math.pi

DIRECTIONS = ['left', 'up', 'right' ,'down']

WAVE_VEL = 0.04


NUM_OSC_FOR_WAVE = 3
SIZE_MUL = 1.45
SIZE_DIFF = 10
ENERGETIC_THRES = 50

MOVEMENT_THRES = 35

class DetectionWindow(object):
    def __init__(self, window_frames=16):
        self.num_frames = window_frames
        self.window = deque([])
        self.detectors = [
            ('jump', self._detect_jump),
            ('wave_double', self._detect_wave_double),
            ('wave_left', self._detect_wave_left),
            ('wave_right', self._detect_wave_right),
            #('sway', self._detect_sway),
            ('energetic', self._detect_energetic),
            ('still', self._detect_still)
        ]
    
    def add_frame(self, values):
        if len(self.window) >= self.num_frames:
            self.window.popleft()
        self.window.append(values)
    
    def detect_event(self):
        event = None
        for detector in self.detectors:
            detector_name, detector_delegate = detector
            is_detected = detector_delegate()
            if is_detected:
                return detector_name

    def _get_num_waves(self, side):
        oscillations = 0
        direction = None

        for frame in self.window:
            roi = frame.get(side, None)
            
            if roi is None:
                continue

            new_dir = roi['direction']
            vx, vy  = roi['velocity']

            fx, fy, fs = frame['head']
            wave_vel = WAVE_VEL * fs

            if roi['n'] > 5 and (vx**2 + vy**2) > wave_vel**2:
                if direction is None:
                    direction = new_dir

                if direction != new_dir and new_dir in ['left', 'right']:
                    oscillations += 1

        return oscillations

    def _movement_x(self):
        start = self.window[0]
        end = self.window[-1]

        start_x, start_y, start_s = start['head']
        end_x, end_y, end_s = end['head']

        return abs(end_y - start_y)


    def _detect_wave_left(self):
        return self._movement_x() < MOVEMENT_THRES and self._get_num_waves('left') > NUM_OSC_FOR_WAVE

    def _detect_wave_right(self):
        return self._movement_x() < MOVEMENT_THRES and self._get_num_waves('right') > NUM_OSC_FOR_WAVE

    def _detect_wave_double(self):
        return self._movement_x() < MOVEMENT_THRES and self._detect_wave_left() and self._detect_wave_right()

    def _detect_jump(self):
        highest = 0
        lowest  = float("inf")
        av_s = 0
        n = 0
        sizes = []
        for frame in self.window:
            fx, fy, fs = frame['head']

            sizes.append(fs)

            av_s += fs
            n += 1

            if fy > highest:
                highest = fy
            elif fy < lowest:
                lowest = fy

        av_s /= n

        largest  = max(sizes)
        smallest = min(sizes)

        diff = (largest - smallest)

        return ((highest - lowest) > (av_s * SIZE_MUL)) and (diff < SIZE_DIFF)

    def _detect_energetic(self):
        energy = 0
        n = 0

        for frame in self.window:
            right = frame.get('right', None)
            left = frame.get('left', None)
            vx, vy = frame.get('head_v', (None, None))

            if right is None or left is None:
                continue

            if right['n'] > 5:
                rvx, rvy = right['velocity']
                energy += (rvx**2 + rvy**2)
                n += 1

            if left['n'] > 5:
                rvx, rvy = left['velocity']
                energy += (rvx**2 + rvy**2)
                n += 1

            energy += vx**2 + vy**2
            n += 1

        if n == 0:
            n = 1

        energy /= n

        return n > 5 and energy > ENERGETIC_THRES

    def _detect_still(self):
        return True

def calc_flow(last_frame, curr_frame):
    frame_h, frame_w = curr_frame.shape
    new_size = (int(frame_w/2), int(frame_h/2))

    last = cv2.resize(last_frame, new_size)
    curr = cv2.resize(curr_frame, new_size)
    return cv2.calcOpticalFlowFarneback(last, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def detect_movement_in_rect(flow, rect, bounds, step=8, threshold=3, resolution=0.5):
    max_w, max_h = bounds
    x1, y1, x2, y2 = rect

    x1 = max(0, x1)
    y1 = max(0, y1)
    w = min(abs(x2 - x1), max_w - x1 - 1)
    h = min(abs(y2 - y1), max_h - y1 - 1)

    rect = (x1, y1, x1 + w, y1 + h)

    y, x = np.mgrid[0:h*resolution:step, 0:w*resolution:step].reshape(2,-1)
    x = (x + x1*resolution).astype(int)
    y = (y + y1*resolution).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x/resolution, y/resolution, (x+fx)/resolution, (y+fy)/resolution]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    threshold = threshold**2  
    lines = [np.array([(x1, y1), (x2, y2)]) for (x1, y1), (x2, y2) in lines if (x2 - x1)**2 + (y2 - y1)**2 > threshold]

    position = 'top'
    velocityX = 0
    velocityY = 0
    centerX = 0
    centerY = 0
    n = 0

    for (line_x1, line_y1), (line_x2, line_y2) in lines:
        velocityX += (line_x2 - line_x1)
        velocityY += (line_y2 - line_y1)

        centerX += line_x1
        centerY += line_y1

        n += 1

    if n == 0:
        n = 1

    centerY /= n
    centerX /= n 

    velocity = np.array((velocityX/n, velocityY/n))

    return {
        'velocity': velocity,
        'center': (centerX, centerY),
        'n': n
    }

def detect_movement_params(flow, rect, bounds, face_v):
    params = detect_movement_in_rect(flow, rect, bounds)

    max_w, max_h = bounds
    x1, y1, x2, y2 = rect

    x1 = max(0, x1)
    y1 = max(0, y1)
    w = min(abs(x2 - x1), max_w - x1 - 1)
    h = min(abs(y2 - y1), max_h - y1 - 1)

    centerX, centerY = params['center']
    velocity = params['velocity']
    n = params['n']

    third_h = h/3

    if centerY < y1 + third_h:
        position = 'top'
    elif centerY < y1 + third_h * 2:
        position = 'mid'
    else:
        position = 'bottom'

    vx, vy = velocity
    f_vx, f_vy = face_v

    vx -= f_vx
    vy -= f_vy

    angle = np.arctan2(vy, vx) + PI

    direction = int(np.round(angle/(TAU/4))) % 4

    return {
        'center': params['center'],
        'position': position,
        'velocity': (vx, vy),
        'direction': DIRECTIONS[direction],
        'n': n,
        'rect': rect
    }


def get_action_region(face):
    x, y, size = face['feature']
    detection_area_x = 3.0 * size
    return (x - detection_area_x, x + detection_area_x, face)

def fix_overlaps(area_a, area_b):
    a_x1, a_x2, a_face = area_a
    b_x1, b_x2, b_face = area_b

    if b_x1 < a_x2:
        # regions are overlapping
        midpoint = (b_x1 + a_x2)/2
        area_a = (a_x1, midpoint, a_face)
        area_b = (midpoint, b_x2, b_face)

    return (area_a, area_b)

def is_interesting(face):
    return face['alive_for'] > 5 and face['matches_made'] < 10 and face.get('has_moved', False)

def get_action_regions(features):
    interesting_regions = []

    for face in features:
        if is_interesting(face):
            # add region to interesting regions
            interesting_regions.append(get_action_region(face))
        else:
            # remove the detector
            face['movement'] = None

    detection_areas = sorted(interesting_regions, key=lambda region: region[0])

    num_areas = len(detection_areas)

    for i in range(num_areas):
        if i < num_areas-1:
            detection_areas[i], detection_areas[i+1] = fix_overlaps(detection_areas[i], detection_areas[i+1])

    return detection_areas

def get_face_velocity(frame, flow, face):
    h, w = frame.shape[:2]
    x, y, face_size = face['feature']
    face_rect = (int(x-face_size/2.0), int(y-face_size/2.0), int(x+face_size/2.0), int(y+face_size/2.0))
    try:
        face_v = detect_movement_in_rect(flow, face_rect, (w,h))['velocity']
    except:
        face_v = (0,0)

    return face_v

def detect_actions(frame, flow, action_regions):
    h, w = frame.shape[:2]

    for region in action_regions:
        left_x, right_x, face = region
        x, y, face_size = face['feature']
        mid_x = x
        top_y = y - (face_size * 2)
        bottom_y = y + (face_size * 2)

        left_rect  = (left_x, top_y, mid_x, bottom_y)
        right_rect = (mid_x, top_y, right_x, bottom_y)

        if 'movement' not in face or face['movement'] is None:
            face['movement'] = DetectionWindow()

        face_v = face['v']

        try:
            face['movement'].add_frame({
                'head': face['feature'],
                'head_v': face_v,
                'left':  detect_movement_params(flow, left_rect, (w,h), face_v),
                'right': detect_movement_params(flow, right_rect, (w,h), face_v)
            })
        except:
            face['movement'].add_frame({
                'head': face['feature'],
                'left': None,
                'right': None
            })

        action = face['movement'].detect_event()
        last_action = face.get('last_movement', 'still')

        if action != last_action:
            face['action'] = last_action
        else:
            face['action'] = action

        face['last_movement'] = action
