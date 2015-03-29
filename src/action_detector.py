import cv2, numpy as np
import math

TAU = math.pi * 2
PI = math.pi


def calc_flow(last_frame, curr_frame):
    frame_h, frame_w = curr_frame.shape
    new_size = (int(frame_w/2), int(frame_h/2))

    last = cv2.resize(last_frame, new_size)
    curr = cv2.resize(curr_frame, new_size)
    return cv2.calcOpticalFlowFarneback(last, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)


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
    unit_x, unit_y = velocity / np.linalg.norm(velocity)

    direction = None
    

    return {
        'position': position,
        'velocity': velocity
    }


def detect_movement(size, flow, features):

    movements = [
        {
            'left': {
                'position': 'top', #'mid', 'bottom'
                'velocity': (x, y)
            },
            'right': {
                # ...
            }
        }
    ]

    return movements


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




