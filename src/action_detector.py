import cv2, numpy as np
import math
from collections import deque

TAU = math.pi * 2
PI = math.pi

DIRECTIONS = ['e','ne','n','nw','w','sw','s','se']

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
            is_detected, params = detector()
            if is_detected:
                return params

    def _detect_wave_left(self):
        return False, {}

    def _detect_wave_right(self):
        return False, {}

    def _detect_wave_double(self):
        return False, {}

    def _detect_jump(self):
        return False, {}

    def _detect_energetic(self):
        return False, {}

    def _detect_still(self):
        return False, {}


def calc_flow(last_frame, curr_frame):
    frame_h, frame_w = curr_frame.shape
    new_size = (int(frame_w/2), int(frame_h/2))

    last = cv2.resize(last_frame, new_size)
    curr = cv2.resize(curr_frame, new_size)
    return cv2.calcOpticalFlowFarneback(last, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def group_flow(flow):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h/2:step, step/2:w/2:step].reshape(2,-1)
    x = x.astype(int)
    y = y.astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x*2, y*2, (x+fx)*2, (y+fy)*2]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    threshold = 8
    threshold = threshold**2

    for line in lines:
        pass
    lines = [np.array([(x1, y1), (x2, y2)]) for (x1, y1), (x2, y2) in lines if (x2 - x1)**2 + (y2 - y1)**2 > threshold]

def detect_movement_params(flow, rect, reference, height, step=8, threshold=5):
    x1, y1, x2, y2 = rect
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    y, x = np.mgrid[step/2:h/2:step, step/2:w/2:step].reshape(2,-1)
    x = x.astype(int)
    y = y.astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x*2, y*2, (x+fx)*2, (y+fy)*2]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    threshold = threshold**2  
    lines = [np.array([(x1, y1), (x2, y2)]) for (x1, y1), (x2, y2) in lines if (x2 - x1)**2 + (y2 - y1)**2 > threshold]

    position = 'top'
    velocityX = 0
    velocityY = 0
    n = 0

    ref_x, ref_y = reference
    for (x1, y1), (x2, y2) in lines:
        velocityX += (x2 - x1)
        velocityY += (y1 - y1)

        if abs(ref_y - y1) < height:
            position = 'mid'
        elif (ref_y - y1) > height:
            position = 'bottom'

        n += 1

    velocity = np.array((velocityX/n, velocityY/n))
    vx, vy = velocity

    angle = np.arctan2(vy, vx);

    direction = np.round(angle/(TAU/8)) * 8

    return {
        'position': position,
        'velocity': velocity,
        'direction': DIRECTIONS[direction]
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
    detection_areas = sorted([get_action_region(face) for face in features if is_interesting(face)], key=lambda region: region[0])

    num_areas = len(detection_areas)

    for i in range(num_areas):
        if i < num_areas-1:
            detection_areas[i], detection_areas[i+1] = fix_overlaps(detection_areas[i], detection_areas[i+1])

    return detection_areas

def detect_actions(size, flow, features, action_regions):
    # TODO
    return features

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




