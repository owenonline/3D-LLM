import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
import json
import argparse
import h5py
from lavis.common.registry import registry

def main():
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--scan_set_name", default="r1_scan", type=str)
    args = parser.parse_args()

    DEVICE = "mps"#"cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "../3DLLM_BLIP2-base/checkpoints/pretrain_blip2_sam_flant5xl_v2.pth"
    print("Loading model from checkpoint...")
    model_cfg = {
        "arch": "blip2_t5",
        "model_type": "pretrain_flant5xl",
        "use_grad_checkpoint": False,
    }
    model_cfg = OmegaConf.create(model_cfg)
    model = registry.get_model_class(model_cfg.arch).from_pretrained(model_type=model_cfg.model_type)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(DEVICE)

    processor_cfg = {"name": "blip_question", "prompt": ""}
    processor_cfg = OmegaConf.create(processor_cfg)
    text_processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)

    feature_path = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "fused_3d_features", "pointclouds", "pc_embeddings.h5")
    points_path = os.path.join(os.getcwd(), "datasets", args.scan_set_name, "fused_3d_features", "pointclouds", "pc_points.h5")

    with h5py.File(feature_path, "r") as f:
        pc_feature = torch.from_numpy(f["pc_embeddings"][:]).to(DEVICE).unsqueeze(0)
    with h5py.File(points_path, "r") as f:
        pc_points = torch.from_numpy(f["pc_points"][:]).to(DEVICE).unsqueeze(0)

    print(pc_feature.shape)
    print(pc_points.shape)

    prompt = "Describe the 3D scene."
    prompt = text_processor(prompt)

    idx = torch.randint(0, pc_feature.shape[1], (13000,))

    pc_feature = pc_feature[:, idx, :]
    pc_points = pc_points[:, idx, :]
    
    model_inputs = {"text_input": prompt, "pc_feat": pc_feature, "pc": pc_points}

    model_outputs = model.predict_answers(
        samples=model_inputs,
        max_len=50,
        length_penalty=1.2,
        repetition_penalty=1.5,
    )
    model_outputs = model_outputs[0]
    print(model_outputs)

if __name__ == "__main__":
    main()