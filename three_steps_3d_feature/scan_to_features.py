import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import open_clip
import torch
import torchvision
from torch import nn
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange
import glob
import argparse
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import tyro
from gradslam.datasets import (
    ICLDataset,
    ReplicaDataset,
    ScannetDataset,
    load_dataset_config,
)
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from typing_extensions import Literal
import json

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512

def create_masks(save_dir, dataset_dir):
    torch.autograd.set_grad_enabled(False)

    sam = sam_model_registry["vit_h"](checkpoint=Path("sam_vit_h_4b8939.pth"))
    sam.to(device="cpu")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    # model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    # model.eval()

    # device = torch.device("cpu")
    
    images_path = dataset_dir + "/*png"
    data_list = glob.glob(images_path)

    for img_name in tqdm(data_list):
        savefile = os.path.join(
            save_dir, os.path.basename(img_name).replace(".png", ".pt")
        )
        if os.path.exists(savefile):
            continue

        imgfile = img_name
        img = cv2.imread(imgfile)
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        _savefile = os.path.join(
                save_dir,
                os.path.splitext(os.path.basename(imgfile))[0] + ".pt",
            )
        
        mask_list = []
        for mask_item in masks:
            mask_list.append(mask_item["segmentation"])

        mask_np = np.asarray(mask_list)
        mask_torch = torch.from_numpy(mask_np)

        torch.save(mask_torch, _savefile)

def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds

def clip_sam(save_dir_path, scene_dir_path, mask_dir_path):
    OPENCLIP_MODEL = "ViT-L-14"  # "ViT-bigG-14"
    OPENCLIP_DATA = "laion2b_s32b_b82k"  # "laion2b_s39b_b160k"
    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL, OPENCLIP_DATA)
    model.visual.output_tokens = True
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    for file in os.listdir(os.path.join(mask_dir_path)):
        INPUT_IMAGE_PATH = os.path.join(scene_dir_path, file.replace(".pt", ".png"))
        SEMIGLOBAL_FEAT_SAVE_FILE = os.path.join(save_dir_path, file)
        if os.path.isfile(SEMIGLOBAL_FEAT_SAVE_FILE):
            continue

        raw_image = cv2.imread(INPUT_IMAGE_PATH)
        raw_image = cv2.resize(raw_image, (512, 512))
        image = torch.tensor(raw_image)

        """
        Extract and save global feat vec
        """
        global_feat = None
        # with torch.cuda.amp.autocast():
        _img = preprocess(Image.open(INPUT_IMAGE_PATH)).unsqueeze(0)  # [1, 3, 224, 224]
        imgfeat = model.visual(_img)[1]  # All image token feat [1, 256, 1024]
        imgfeat = torch.mean(imgfeat, dim=1)

        global_feat = imgfeat.half()#.cuda()

        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        FEAT_DIM = global_feat.shape[-1]

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        MASK_LOAD_FILE = os.path.join(mask_dir_path, file)
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, FEAT_DIM, dtype=torch.half)

        mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
        num_masks = mask.shape[-3]

        rois = []
        roi_similarities_with_global_vec = []
        roi_sim_per_unit_area = []
        feat_per_roi = []
        roi_nonzero_inds = []

        for _i in range(num_masks):
            curmask = mask[0, _i]
            bbox, nonzero_inds = get_bbox_around_mask(curmask)
            x0, y0, x1, y1 = bbox

            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
            iou = bbox_area / img_area

            if iou < 0.005:
                continue
            with torch.no_grad():
                img_roi = image[x0:x1, y0:y1]
                img_roi = Image.fromarray(img_roi.numpy())
                img_roi = preprocess(img_roi).unsqueeze(0)#.cuda()
                roifeat = model.visual(img_roi)[1]  # All image token feat [1, 256, 1024]
                roifeat = torch.mean(roifeat, dim=1)

                feat_per_roi.append(roifeat)
                roi_nonzero_inds.append(nonzero_inds)
                _sim = cosine_similarity(global_feat, roifeat)

                rois.append(torch.tensor(list(bbox)))
                roi_similarities_with_global_vec.append(_sim)
                roi_sim_per_unit_area.append(_sim)

        rois = torch.stack(rois)
        scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
        retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
        feat_per_roi = torch.cat(feat_per_roi, dim=0)  # N, 1024

        retained_rois = rois[retained]
        retained_scores = scores[retained]
        retained_feat = feat_per_roi[retained]
        retained_nonzero_inds = []
        for _roiidx in range(retained.shape[0]):
            retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])

        mask_sim_mat = torch.nn.functional.cosine_similarity(
            retained_feat[:, :, None], retained_feat.t()[None, :, :]
        )
        mask_sim_mat.fill_diagonal_(0.0)
        mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
        softmax_scores = retained_scores - mask_sim_mat
        softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)
        for _roiidx in range(retained.shape[0]):
            _weighted_feat = (
                softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
            )
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += (
                _weighted_feat[0].detach().cpu().half()
            )
            outfeat[
                retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]
            ] = torch.nn.functional.normalize(
                outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(),
                dim=-1,
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim

        torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)

def fuse_features(save_dir, dataset_dir, depth_map_dir, multiview_feat_dir, pose_dir, intrinsics_path, ds_info_path):
    slam = PointFusion(odom="gt", dsratio=1, device="mps", use_embeddings=True)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(
        device="cpu",
    )

    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()

    ds_info = json.load(open(ds_info_path))

    for i in trange(10):#ds_info['num_views']):
        color_image = cv2.imread(os.path.join(dataset_dir, f"{i}_rgb.png"))
        color_image = cv2.resize(color_image, (512, 512))

        depth_image = np.load(os.path.join(depth_map_dir, f"{i}_depth.npy"))
        depth_image = cv2.resize(depth_image, (512, 512))

        _color = torch.from_numpy(color_image).float()
        _depth = torch.from_numpy(depth_image).float().unsqueeze(-1)
        _pose = torch.from_numpy(np.load(os.path.join(pose_dir, f"{i}_pose.npy"))).float()
        _embedding = torch.load(os.path.join(multiview_feat_dir, f"{i}_rgb.pt"))
        _embedding = _embedding.float()
        _embedding = torch.nn.functional.interpolate(
            _embedding.permute(2, 0, 1).unsqueeze(0).float(),
            [512, 512],
            mode="nearest",
        )[0].permute(1, 2, 0).half()

        # print(_color.shape, _depth.shape, _pose.shape, _embedding.shape)

        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0),
            _pose.unsqueeze(0).unsqueeze(0),
            embeddings=_embedding.unsqueeze(0).unsqueeze(0),
        )
        
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)
        # frame_prev = frame_cur

    pointclouds.save_to_h5(save_dir)

def main():
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--scan_set_name", default="r1_scan", type=str)
    args = parser.parse_args()
    dataset_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "rgb_images")

    mask_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "sam_masks")
    os.makedirs(mask_dir, exist_ok=True)
    # create_masks(mask_dir, dataset_dir)

    multiview_feat_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "multiview_features")
    os.makedirs(multiview_feat_dir, exist_ok=True)
    # clip_sam(multiview_feat_dir, dataset_dir, mask_dir)

    fused_3d_feat_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "fused_3d_features")
    os.makedirs(fused_3d_feat_dir, exist_ok=True)
    depth_map_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "depth_maps")
    pose_dir = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "poses")
    intrinsics_path = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "intrinsics.npy")
    ds_info_path = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "dataset_info.json")
    fuse_features(fused_3d_feat_dir, dataset_dir, depth_map_dir, multiview_feat_dir, pose_dir, intrinsics_path, ds_info_path)

if __name__ == "__main__":
    main()


    


    

    

    