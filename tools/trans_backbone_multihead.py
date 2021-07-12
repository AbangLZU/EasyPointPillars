import torch
from torch import nn
import numpy as np
from onnx_utils.onnx_backbone_2d import BaseBEVBackbone
# from onnx_utils.onnx_dense_head import  AnchorHeadMulti, SingleHead
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle
from pcdet.config import cfg, cfg_from_yaml_file

import argparse

class backbone(nn.Module):
    def __init__(self, cfg , gridx , gridy):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
        self.dense_head =  AnchorHeadSingle(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels=384,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=np.array([gridx , gridy , 1]),
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        batch_cls_preds, batch_box_preds = self.dense_head.forward(x)

        return batch_cls_preds, batch_box_preds


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
    parser.add_argument('--output_path', type=str, default='../output/kitti_models/pointpillar/onnx/backbone_head.onnx', 
                        help='specify the onnx backbone model output path')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import numpy as np 
    from pcdet.config import cfg, cfg_from_yaml_file

    args = parse_config()
    # cfg_file = '/path/to/cbgs_pp_multihead.yaml'
    # filename_mh = "/path/to/pp_multihead_nds5823_updated.pth"
    cfg_file = args.cfg_file
    filename_mh = args.ckpt

    cfg_from_yaml_file(cfg_file, cfg)
    model , dummy_input = build_backbone_multihead(filename_mh , cfg )

    # export_onnx_file = "/path/to/cbgs_pp_multihead_backbone.onnx"
    export_onnx_file = args.output_path
    
    model.eval().cuda()
    torch.onnx.export(model,
                      dummy_input,
                      export_onnx_file,
                      opset_version=10,
                      verbose=True,
                      do_constant_folding=True) # 输出名
