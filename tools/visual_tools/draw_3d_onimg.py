from tools.visual_tools.calib import Calibration
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from matplotlib.lines import Line2D

colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
names = ['Car', 'Pedestrian', 'Cyclist']


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def draw_line(p1, p2, front, lab):
    for item in [p1[0], p2[0], p1[1], p2[1]]:
        if item < 0:
            return
    plt.gca().add_line(Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[int(lab-1)]))


def draw_box_onimage(img_path, label_path, calib_path):
    # load image
    img = np.array(io.imread(img_path), dtype=np.int32)

    # load labels
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # load calibration file
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    fig = plt.figure()
    # draw image
    plt.imshow(img)

    for line in labels:
        line = line.split()
        x, y, z, l, w, h, rot, lab = line # rot is z axis

        h, w, l, x, y, z, rot, lab = map(float, [h, w, l, x, y, z, rot, lab])
        # filter the label probability < 0.3
        if lab >= 0:
            # 3d bounding box corners
            Box = np.array(
                [
                    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
                ]
            )
            # the output rotation is z axis
            R = rotz(rot)
            corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)

            corners_3d[0, :] = corners_3d[0, :] + x
            corners_3d[1, :] = corners_3d[1, :] + y
            corners_3d[2, :] = corners_3d[2, :] + z

            box_corners = np.transpose(corners_3d)
            #         7 -------- 6
            #        /|         /|
            #       4 -------- 5 .
            #       | |        | |
            #       . 3 -------- 2
            #       |/         |/
            #       0 -------- 1

            calib = Calibration(calib_path)

            ## project box from velo to rect
            box_corners = calib.project_velo_to_rect(box_corners)

            n = box_corners.shape[0]
            pts_3d_extend = np.hstack((box_corners, np.ones((n, 1))))
            # pts_2d: nx2 matrix
            pts_2d = np.dot(pts_3d_extend, np.transpose(P2))  # nx3
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            corners_img = pts_2d[:, 0:2]

            # draw the upper 4 horizontal lines
            draw_line(corners_img[0], corners_img[1], 0, lab)  # front = 0 for the front lines
            draw_line(corners_img[1], corners_img[2], 1, lab)
            draw_line(corners_img[2], corners_img[3], 1, lab)
            draw_line(corners_img[3], corners_img[0], 1, lab)

            # draw the lower 4 horizontal lines
            draw_line(corners_img[4], corners_img[5], 0, lab)
            draw_line(corners_img[5], corners_img[6], 1, lab)
            draw_line(corners_img[6], corners_img[7], 1, lab)
            draw_line(corners_img[7], corners_img[4], 1, lab)

            # draw the 4 vertical lines
            draw_line(corners_img[4], corners_img[0], 0, lab)
            draw_line(corners_img[5], corners_img[1], 0, lab)
            draw_line(corners_img[6], corners_img[2], 1, lab)
            draw_line(corners_img[7], corners_img[3], 1, lab)

    # fig.patch.set_visible(False)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    draw_box_onimage('../../data/kitti/testing/image_2/000099.png', 
        'predicted.txt', '../../data/kitti/testing/calib/000099.txt')