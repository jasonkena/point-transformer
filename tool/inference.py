import os
import sys
import torch

sys.path.append("/data/adhinart/freseg")

from freseg_inference import evaluation
from train import weird_collate
import argparse

from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
from collections import OrderedDict


def weird_collate(points):
    # see https://github.com/POSTECH-CVLab/point-transformer/blob/10d43ab5210fc93ffa15886f2a4c6460cc308780/util/data_util.py#L16C1-L23C88
    # transforms points: [B, N, 3] -> [B*N, 3]
    # outputs offsets 
    B, N, _ = points.shape
    coord = points.reshape(-1, 3).float()
    feat = coord
    offset = torch.arange(1, B+1) * N 
    offset = offset.to(points.device).int()

    return coord, feat, offset

def load_model(output_dir):
    fea_dim = 3
    num_classes = 2
    model = Model(c=fea_dim, k=num_classes).cuda()

    state_dict = torch.load(os.path.join(output_dir, "model", "model_best.pth"))["state_dict"]
    
    # since the model was trained using multiple GPUs, the keys in the state_dict have 'module.' in them
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' if it exists
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    return model

def model_inference(model, points):
    coord, feat, offset = weird_collate(points) # (n, 3), (n, c), (n), (b)

    output = model([coord, feat, offset])
    pred_max = output.max(1)[1]

    return pred_max

def main(args):
    exp_dir = f"./exp/freseg/{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    assert os.path.exists(exp_dir), f"Experiment {exp_dir} does not exist"

    evaluation(
        output_path=exp_dir,
        fold=args.fold,
        path_length=args.path_length,
        npoint=args.npoint,
        frenet=args.frenet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_model=load_model,
        model_inference=model_inference
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch Size during training"
    )
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument("--path_length", type=int, help="path length")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--num_workers", type=int, default=16, help="num workers")

    parser.add_argument(
        "--frenet", action="store_true", help="whether to use Frenet transformation"
    )

    args = parser.parse_args()

    main(args)
