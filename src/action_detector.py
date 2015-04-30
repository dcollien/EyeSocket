import cv2, numpy as np
import math
from collections import deque

TAU = math.pi * 2
PI = math.pi

DIRECTIONS = ['<<', '^^', '>>' ,'vv']

# This module is a work in progress

class DetectionWindow(object):
    def __init__(self, window_frames=16):
        self.num_frames = window_frames
        self.window = deque([])
        self.detectors = [
            ('wave_double', self._detect_wave_double),
            ('wave_left', self._detect_wave_left),
            ('wave_right', self._detect_wave_right),
            ('jump', self._detect_jump),
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

    def _detect_wave_left(self):
        return False

    def _detect_wave_right(self):
        return False

    def _detect_wave_double(self):
        return False

    def _detect_jump(self):
        return False

    def _detect_energetic(self):
        return False

    def _detect_still(self):
        return True

def calc_flow(last_frame, curr_frame):
    frame_h, frame_w = curr_frame.shape
    new_size = (int(frame_w/2), int(frame_h/2))

    last = cv2.resize(last_frame, new_size)
    curr = cv2.resize(curr_frame, new_size)
    return cv2.calcOpticalFlowFarneback(last, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def detect_movement_in_rect(flow, rect, bounds, step=8, threshold=5, resolution=0.5):
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
    detection_area = 2.3 * size
    return (x - detection_area, x + detection_area, face)

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
    return face['alive_for'] > 5 and face['matches_made'] < 10

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

        face_rect = (int(x-face_size/2.0), int(y-face_size/2.0), int(x+face_size/2.0), int(y+face_size/2.0))
        face_v = detect_movement_in_rect(flow, face_rect, (w,h))['velocity']

        face['movement'].add_frame({
            'head': face['feature'],
            'left':  detect_movement_params(flow, left_rect, (w,h), face_v),
            'right': detect_movement_params(flow, right_rect, (w,h), face_v)
        })

        face['action'] = face['movement'].detect_event()


"""
# trying to infer pose from a skeleton template
def get_dimensions(head):
    head_x, head_y, head_h = (int(head[0]), int(head[1]), int(head[2]))

    return {
        'neck': int(head_h * 0.4),
        'shoulder': int(head_h * 1.1),
        'forearm': int(head_h * 1.8)
    }

def sample_line(flow, start, angle, length, resolution=0.5):
    sample_sum = 0
    pts_sampled = 0
    for i in range(100):
        sample_length = math.randint(0, length)
        sample_x = math.cos(angle) * sample_length
        sample_y = math.sin(angle) * sample_length
        flow_y, flow_x = flow[int(sample_y * resolution), int(sample_x * resolution)]
        sample_sum += (flow_y + flow_x)/2.
        pts_sampled += 1

    return sample_sum/pts_sampled

def infer_pose(flow, head):
    dimensions = get_dimensions(head)

    head_x, head_y, head_h = head
    head_x, head_y, head_h = int(head_x), int(head_y), int(head_h)

    neck_length = dimensions['neck']
    shoulder_length = dimensions['shoulder']
    forearm_length  = dimensions['forearm']

    chin       = (head_x, int(head_y + head_h/2.))
    chest      = (chin[0], chin[1] + neck_length)
    l_shoulder = (chest[0] - shoulder_length, chest[1])
    r_shoulder = (chest[0] + shoulder_length, chest[1])

    segments = 16
    sweep = TAU/4

    l_best_angle = sweep
    l_best_value = 0    
    r_best_angle = sweep
    r_best_value = 0

    for i in range(segments):
        sweep += PI/segments
        l_sample = sample_line(flow, l_shoulder, sweep, shoulder_length)
        r_sample = sample_line(flow, r_shoulder, sweep, shoulder_length)
        if l_sample > l_best_value:
            l_best_angle = sweep
            l_best_value = l_sample
        if r_sample > r_best_value:
            r_best_angle = sweep
            r_best_value = r_sample

    lx, ly = l_shoulder

    return {
        'l_forearm': l_best_value * shoulder_length
    }
"""


