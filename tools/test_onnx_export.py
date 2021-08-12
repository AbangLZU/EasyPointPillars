import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate
from pcdet.datasets import DatasetTemplate
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

import torch
from torch import nn
import numpy as np

from onnx_utils.onnx_backbone_2d import BaseBEVBackbone
from onnx_utils.onnx_single_head import  AnchorHeadSingle
from visual_tools import draw_clouds_with_boxes

from pcdet.config import cfg, cfg_from_yaml_file
import argparse


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


class backbone(nn.Module):
    def __init__(self, cfg , gridx , gridy):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
        self.dense_head =  AnchorHeadSingle(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels=128*3,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=np.array([gridx , gridy , 1]),
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        batch_cls_preds, batch_box_preds = self.dense_head.forward(x)

        return batch_cls_preds, batch_box_preds

# adapted from pcdet/models/backbones_3d/vfe/pillar_vfe.py
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.part = 50000

    def forward(self, inputs):
        # nn.Linear performs randomly when batch size is too large
        x = self.linear(inputs)

        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        
        # if self.with_distance:
        #     num_point_features += 1
        self.num_point_features = num_point_features
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, features, **kwargs):
  
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features[:,0,:]
        return features


def build_pfe(ckpt,cfg):

    pfe =PillarVFE(            
                model_cfg=cfg.MODEL.VFE,
                num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,  
                voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)  

    pfe.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "vfe" in key:
            dicts[key[4:]] = checkpoint["model_state"][key]
    pfe.load_state_dict(dicts)
    
    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
    max_points_per_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
    dims_feature = pfe.num_point_features
    dummy_input = torch.ones(max_num_pillars,max_points_per_pillars,dims_feature).cuda()
    return pfe , dummy_input 

def build_backbone_multihead(ckpt , cfg ):

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    grid_size = (pc_range[3:] - pc_range[:3]) /voxel_size
    gridx = grid_size[0].astype(np.int)
    gridy = grid_size[1].astype(np.int)
    model = backbone(cfg , gridx ,gridy)
    model.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    model.load_state_dict(dicts)

    dummy_input = torch.ones(1, 64, gridx, gridy).cuda()
    return model , dummy_input


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth', 
                        help='specify the pretrained model')
    parser.add_argument('--data_path', type=str, default='../data/kitti/testing/velodyne/000099.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    
    parser.add_argument('--output_path', type=str, default='../output/kitti/pfe.onnx', 
                        help='specify the onnx pfe model output path')

    args = parser.parse_args()
    return args


def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

def preprocess_vfe(batch_dict, model_cfg):

    # read config params
    use_absolute_xyz = model_cfg.USE_ABSLOTE_XYZ
    with_distance =  model_cfg.WITH_DISTANCE
    voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    voxel_x = voxel_size[0]
    voxel_y = voxel_size[1]
    voxel_z = voxel_size[2]
    x_offset = voxel_x / 2 + point_cloud_range[0]
    y_offset = voxel_y / 2 + point_cloud_range[1]
    z_offset = voxel_z / 2 + point_cloud_range[2]

    voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
    points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * voxel_x + x_offset)
    f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * voxel_y + y_offset)
    f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * voxel_z + z_offset)

    if use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]

    if with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
    features *= mask
    return features

if __name__ == "__main__":
    from pcdet.config import cfg, cfg_from_yaml_file
    args = parse_config()

    logger = common_utils.create_logger()

    cfg_file = args.cfg_file
    filename_mh = args.ckpt
    cfg_from_yaml_file(cfg_file, cfg)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    model_cfg=cfg.MODEL
    model_pfe , dummy_input  = build_pfe( filename_mh, cfg)
    print('input to pfe size: ', dummy_input.cpu().shape)

    model_backbone_head , dummy_input2 = build_backbone_multihead(filename_mh , cfg)
    print('input to backbone size: ', dummy_input2.cpu().shape)

    model_pfe.eval().cuda()
    model_backbone_head.eval().cuda()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            vfe_features = preprocess_vfe(data_dict, model_cfg=cfg.MODEL.VFE)
            print(vfe_features.shape)

            # the input to the pfe should be the extracted voxel features:[40000, 32, 10]
            # which is [num_voxels, max_points_per_voxel, 3 + C]
            pred_dicts, _ = model_pfe.forward(vfe_features)

            # points=data_dict['points'][:, 1:]
            # ref_boxes=pred_dicts[0]['pred_boxes']
            # ref_scores=pred_dicts[0]['pred_scores']
            # ref_labels=pred_dicts[0]['pred_labels']

            # points = points.cpu().numpy()
            # boxs = ref_boxes.cpu().numpy()
            # labels = ref_labels.cpu().numpy()
            # scores = ref_scores.cpu().numpy()

            # scores = scores > 0.4
            # for i, sc in enumerate(scores):
            #     if(sc):
            #         labels[i] = labels[i]
            #     else:
            #         labels[i] = -1
            # boxs_with_label = np.c_[boxs, labels.T]
            # # save the detected result for plot on image
            # np.savetxt('../visual_tools/predicted.txt', boxs_with_label)
            # draw_clouds_with_boxes(points, boxs, labels)