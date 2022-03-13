import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from models.psp import pSp
from utils.img_utils import preproc, postproc
from utils.load_utils import load_network_pkl, blend_models, blend_models_stylegan3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs", type=Path, nargs='+', help="Name of image files to be translated")
    parser.add_argument("--pretrain_weight", type=Path, default="stylegan2-pretrained.pkl", 
        help="Pretrained StyleGAN Generator to act as high resolution layer components")
    parser.add_argument("--finetune_weight", type=Path, default="stylegan2-finetuned.pkl", 
        help="Finetuned StyleGAN Generator to act as low resolution layer components")
    parser.add_argument("--e4e_ckpt", type=Path, default="lhq-e4e-encoder.pt")
    parser.add_argument("--network_type", type=str, default="stylegan2", choices=["stylegan2", "stylegan3"], 
        help="Which type of stylegan to use.")
    parser.add_argument("--blend_lv", type=int, default=8, 
        help="If StyleGAN2 : Image resolution for style blending.\n If StyleGAN3 : Network layer for style blending.")
    parser.add_argument("--output_path", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    pretrain_path = Path("weights").joinpath(args.pretrain_weight)
    finetune_path = Path("weights").joinpath(args.finetune_weight)
    e4e_ckpt_path = Path("weights").joinpath(args.e4e_ckpt)

    if args.output_path is None:
        args.output_path = Path(f"imgs/{pretrain_path.stem}_{finetune_path.stem}_{args.blend_lv}")
    args.output_path.mkdir(parents=True, exist_ok=True)

    with open(pretrain_path, 'rb') as f:
        G_pretrain = load_network_pkl(f)['G_ema'].to("cuda")
    with open(finetune_path, 'rb') as f:
        G_finetune = load_network_pkl(f)['G_ema'].to("cuda")

    if args.network_type == "stylegan2":
        G_blended = blend_models(G_pretrain, G_finetune, args.blend_lv)
    else:
        G_blended = blend_models_stylegan3(G_pretrain, G_finetune, args.blend_lv)

    opts = torch.load(e4e_ckpt_path, map_location="cpu")["opts"]
    opts['checkpoint_path'] = e4e_ckpt_path
    opts['device'] = "cuda"
    opts['model_type'] = args.network_type
    net = pSp(argparse.Namespace(**opts)).eval().to("cuda")

    for img_path in tqdm(args.imgs):
        img = Image.open(img_path).resize((256,256))
        img_t = preproc(img).to("cuda")

        with torch.no_grad():
            _, ws = net(img_t, randomize_noise=False, return_latents=True)
            img_pretrain = G_pretrain.synthesis(ws, noise_mode="const")
            img_finetune = G_finetune.synthesis(ws, noise_mode="const")
            img_blended = G_blended.synthesis(ws, noise_mode="const")

        final_img = np.concatenate([postproc(img_pretrain[0]), postproc(img_finetune[0]), postproc(img_blended[0])], axis=1)
        Image.fromarray(final_img).save(args.output_path.joinpath(img_path.name))




