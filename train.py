import os
import re
import json
import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

from utils.collate import CollateFn
from utils.timed import timed_hoc
from gnn import PointNetPP


def make_data_loaders(args):
    dataset_kwargs = {
        "pc_range": args.pc_range,
        "voxel_size": args.voxel_size,
        "n_input": args.n_input,
        "input_step": args.input_step,
        "n_output": args.n_output,
        "output_step": args.output_step,
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": False,
        "drop_last": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    if args.dataset.lower() == "nuscenes":
        from utils.nusc import nuScenesDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(args.nusc_version, args.nusc_root)
        data_loaders = {
            "train": DataLoader(
                nuScenesDataset(nusc, "mini_train", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                nuScenesDataset(nusc, "mini_val", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
        }

    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    return data_loaders


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d, exist_ok=True)


def resume_from_ckpts(ckpt_dir, model, optimizer, scheduler):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Resume training from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = 1 + checkpoint["epoch"]
        n_iter = checkpoint["n_iter"]
    else:
        start_epoch = 0
        n_iter = 0
    return start_epoch, n_iter


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()

    if device_count and args.batch_size % device_count != 0:
        raise RuntimeError(
            f"Batch size ({args.batch_size}) cannot be divided by device count ({device_count})"
        )

    # dataset
    data_loaders = make_data_loaders(args)

    ForecastingNetwork = PointNetPP(num_coords=3, batch_size=args.batch_size)  # 3 coordinates

    model = ForecastingNetwork
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_epoch, gamma=args.lr_decay
    )

    # dump config
    mkdir_if_not_exists(args.model_dir)
    with open(f"{args.model_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # resume
    ckpt_dir = f"{args.model_dir}/ckpts"
    mkdir_if_not_exists(ckpt_dir)

    # data parallel
    model = nn.DataParallel(model)

    writer = SummaryWriter(f"{args.model_dir}/tf_logs")

    for epoch in range(0, args.num_epoch):
        print("Epoch:", epoch)

        for phase in ["train"]:  # , "val"]:
            data_loader = data_loaders[phase]

            if phase == "train":
                # model.train()
                timed_hoc(model.train)
            else:
                model.eval()

            num_batch = len(data_loader)
            # num_example = len(data_loader.dataset)
            for i, batch in enumerate(data_loader):
                print("Batch:", i)

                input_points, input_tindex = batch[1:3]
                _, output_points, output_tindex = batch[3:6]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # out = model(input_points, input_tindex)
                    out = timed_hoc(model, input_points, input_tindex)

                    # https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
                    # loss, _ = chamfer_distance(out, output_points, single_directional=True)
                    loss, _ = timed_hoc(chamfer_distance, out, output_points, single_directional=True)

                    if phase == "train":
                        optimizer.step()
                    avg_loss = loss.mean(axis=0)

                    # Writing the loss function to tensorboard
                    if phase == "train":
                        writer.add_scalar("Loss/train", avg_loss.item(), epoch * len(data_loader) + i)
                    else:
                        writer.add_scalar("Loss/val", avg_loss.item(), epoch * len(data_loader) + i)

                print(
                    f"Phase: {phase},",
                    f"Epoch: {epoch}/{args.num_epoch},",
                    f"Batch: {i}/{num_batch},",
                    f"{loss} Loss: {avg_loss.item():.3f}",
                )

                '''    
                if phase == "train":
                    n_iter += 1
                    writer.add_scalar(f"Train loss:", loss.item(), n_iter)

                if phase == "train" and (i + 1) % (num_batch // 10) == 0:
                    ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}_iter_{n_iter}.pth"
                    torch.save(
                            {
                                "epoch": epoch,
                                "n_iter": n_iter,
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            ckpt_path,
                            _use_new_zipfile_serialization=False,
                    )
            
            if phase == "train":
                ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}.pth"
                torch.save(
                        {
                            "epoch": epoch,
                            "n_iter": n_iter,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        ckpt_path,
                        _use_new_zipfile_serialization=False,
                )
            else:
                for key in total_val_loss:
                    mean_val_loss = total_val_loss[key] / num_example
                    writer.add_scalar(f"{phase}/{key}", mean_val_loss, n_iter)
            '''
        scheduler.step()
    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset", type=str, default="nuscenes")
    data_group.add_argument("--nusc-root", type=str, default="data/sets/nuscenes")
    data_group.add_argument("--nusc-version", type=str, default="v1.0-mini")
    data_group.add_argument(
        "--pc-range",
        type=float,
        nargs="+",
        default=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
    )
    data_group.add_argument("--voxel-size", type=float, default=0.2)
    data_group.add_argument("--n-input", type=int, default=2)
    data_group.add_argument("--input-step", type=int, default=1)
    data_group.add_argument("--n-output", type=int, default=2)
    data_group.add_argument("--output-step", type=int, default=1)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-dir", type=str, required=True)
    model_group.add_argument("--optimizer", type=str, default="Adam")  # Adam with 5e-4
    model_group.add_argument("--lr-start", type=float, default=5e-4)
    model_group.add_argument("--lr-epoch", type=float, default=5)
    model_group.add_argument("--lr-decay", type=float, default=0.1)
    model_group.add_argument("--num-epoch", type=int, default=1)
    model_group.add_argument("--batch-size", type=int, default=4)
    model_group.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    train(args)

