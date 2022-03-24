import cv2
import sys
import torch
import json
import torchvision.transforms as transforms
from lib.hrnet.lib.utils.transforms import *

from lib.hrnet.lib.utils.coco_h36m import coco_h36m
import numpy as np

joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
               [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
               [5, 11], [6, 12], [11, 12],
               [11, 13], [12, 14], [13, 15], [14, 16]]

h36m_pairs = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
              (12, 13), (8, 14), (14, 15), (15, 16)]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255]]


def plot_keypoint(image, coordinates, confidence, keypoint_thresh=0.3):
    # USE cv2
    joint_visible = confidence[:, :, 0] > keypoint_thresh
    coordinates = coco_h36m(coordinates)
    for i in range(coordinates.shape[0]):
        pts = coordinates[i]

        for joint in pts:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 8, (255, 255, 255), 1)

        for color_i, jp in zip(colors, h36m_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt0 = pts[jp, 0]
                pt1 = pts[jp, 1]
                pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])

                cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 6)
                #  cv2.circle(image,(pt0_0, pt0_1), 2, color_i, thickness=-1)
                #  cv2.circle(image,(pt1_0, pt1_1), 2, color_i, thickness=-1)
    return image


def write(x, img):
    x = [int(i) for i in x]
    c1 = tuple(x[0:2])
    c2 = tuple(x[2:4])

    color = [0, 97, 255]
    label = 'People {}'.format(x[-1])
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, [0, 128, 255], -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def load_json(file_path):
    with open(file_path, 'r') as fr:
        video_info = json.load(fr)

    label = video_info['label']
    label_index = video_info['label_index']

    num_frames = video_info['data'][-1]['frame_index']
    keypoints = np.zeros((2, num_frames, 17, 2), dtype=np.float32)  # (M, T, N, 2)
    scores = np.zeros((2, num_frames, 17), dtype=np.float32)  # (M, T, N)

    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']

        for index, skeleton_info in enumerate(frame_info['skeleton']):
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            bbox = skeleton_info['bbox']

            if len(bbox) == 0 or index+1 > 2:
                continue

            pose = np.asarray(pose, dtype=np.float32)
            score = np.asarray(score, dtype=np.float32)
            score = score.reshape(-1)

            keypoints[index, frame_index-1] = pose
            scores[index, frame_index-1] = score

    new_kpts = []
    for i in range(keypoints.shape[0]):
        kps = keypoints[i]
        if np.sum(kps) != 0.:
            new_kpts.append(kps)

    new_kpts = np.asarray(new_kpts, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[:, :, :, np.newaxis]
    return new_kpts, scores, label, label_index


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : (x1, y1, x2, y2)
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1

    center[0] = x1 + box_width * 0.5
    center[1] = y1 + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


# Pre-process
def PreProcess(image, bboxs, cfg, num_pos=2):
    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    else:
        data_numpy = image

    inputs = []
    centers = []
    scales = []

    for bbox in bboxs[:num_pos]:
        c, s = box_to_center_scale(bbox, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input = transform(input).unsqueeze(0)
        inputs.append(input)

    inputs = torch.cat(inputs)
    return inputs, data_numpy, centers, scales
