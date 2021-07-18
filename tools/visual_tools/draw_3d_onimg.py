import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io
from matplotlib.lines import Line2D

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
file_id = '000099'

def draw_line(p1, p2, front=1):
    plt.gca().add_line(Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[names.index(lab) * 2 + front]))


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
        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
        if lab != 'DontCare':
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

            # transform the 3d bbox from object coordiante to camera_0 coordinate
            R = np.array([[np.cos(rot), 0, np.sin(rot)],
                        [0, 1, 0],
                        [-np.sin(rot), 0, np.cos(rot)]])
            corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

            # transform the 3d bbox from camera_0 coordinate to camera_x image
            corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
            corners_img = np.matmul(corners_3d_hom, P2.T)
            corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]


            # draw the upper 4 horizontal lines
            draw_line(corners_img[0], corners_img[1], 0)  # front = 0 for the front lines
            draw_line(corners_img[1], corners_img[2])
            draw_line(corners_img[2], corners_img[3])
            draw_line(corners_img[3], corners_img[0])

            # draw the lower 4 horizontal lines
            draw_line(corners_img[4], corners_img[5], 0)
            draw_line(corners_img[5], corners_img[6])
            draw_line(corners_img[6], corners_img[7])
            draw_line(corners_img[7], corners_img[4])

            # draw the 4 vertical lines
            draw_line(corners_img[4], corners_img[0], 0)
            draw_line(corners_img[5], corners_img[1], 0)
            draw_line(corners_img[6], corners_img[2])
            draw_line(corners_img[7], corners_img[3])

    # fig.patch.set_visible(False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('examples/kitti_3dbox_to_img.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    draw_box_onimage('../../data/kitti/training/image_2/000052.png', 
        '../../data/kitti/training/label_2/000052.txt', '../../data/kitti/training/calib/000052.txt')
  