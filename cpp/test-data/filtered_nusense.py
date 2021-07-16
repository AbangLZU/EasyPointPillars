import os
import numpy as np
import argparse


# def parse_args():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--input_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
#                         help='specify the config for demo')
#     parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/000099.bin',
#                         help='specify the point cloud data file or directory')
#     parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth', 
#                         help='specify the pretrained model')
#     parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

#     args = parser.parse_args()

#     return args, cfg


def load_kitti(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


if __name__ == "__main__":
    points = load_kitti('/media/data/OpenPCDet/cpp/test-data/99.bin')
    print(len(points))
    np.savetxt('/media/data/OpenPCDet/cpp/test-data/99.txt', points)

