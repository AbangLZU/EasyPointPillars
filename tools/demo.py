import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import open3d as o3d
from open3d.visualization import draw_geometries
from visual_tools import draw_clouds_with_boxes
import datetime

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/000099.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth', 
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            start_time = datetime.datetime.now()

            pred_dicts, _ = model.forward(data_dict)

            end_time = datetime.datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            print("Inference 1 sample time ", inference_time, 's')

            points=data_dict['points'][:, 1:]
            ref_boxes=pred_dicts[0]['pred_boxes']
            ref_scores=pred_dicts[0]['pred_scores']
            ref_labels=pred_dicts[0]['pred_labels']
            # print(type(points), points.shape)

            points = points.cpu().numpy()
            boxs = ref_boxes.cpu().numpy()
            labels = ref_labels.cpu().numpy()
            scores = ref_scores.cpu().numpy()
            # print(labels)
            # filter score < 0.4
            scores = scores > 0.4
            for i, sc in enumerate(scores):
                if(sc):
                    labels[i] = labels[i]
                else:
                    labels[i] = -1
            # labels = labels & (scores > 0.4)
            # kitti dataset, 1: Car, 2: Pedestrian, 3: Cyclist
            # print(labels)
            boxs_with_label = np.c_[boxs, labels.T]
            # save the detected result for plot on image
            np.savetxt('visual_tools/predicted.txt', boxs_with_label)
            draw_clouds_with_boxes(points, boxs, labels)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
