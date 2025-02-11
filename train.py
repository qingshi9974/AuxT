import sys
import random
import argparse
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.Meter import AverageMeterTEST, AverageMeterTRAIN
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from models import TCM_AUXT


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out



class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

  

    
def test_epoch(iterations, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeterTEST()
    bpp_loss = AverageMeterTEST()
    mse_loss = AverageMeterTEST()
    psnr_loss = AverageMeterTEST()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_losss=out_criterion["mse_loss"]
            psnr = 10 * (torch.log(1/ mse_losss) / np.log(10))
            mse_loss.update(mse_loss)
            psnr_loss.update(psnr)
      

    print(
        f"\tIterations {iterations}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR loss: {psnr_loss.avg :.3f} |"
        f"\tMSE loss: {mse_loss.avg :.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
      
    )
    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename[:-8]+"_best"+filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--lmbda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=1e-1,
        help="orthogonal penalty parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--test", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="kodak", transform=test_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net = TCM_AUXT(N=64) #TCM_small + AUXT
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    last_epoch = 0
    iterations = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading Checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"]) 
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    if args.test:
        print(len(test_dataloader))
        test_epoch(0, test_dataloader, net, criterion)
        exit(-1)

    para = sum([np.prod(list(p.size())) for p in net.parameters()])/1e6
    print("Number of Parameters",para)
    elapsed,data_time, losses, psnrs, bpps, mse_losses,aux_losses = [AverageMeterTRAIN(200) for _ in range(7)]

    best_loss = float("inf")
    
    for epoch in range(last_epoch, args.epochs):

        net.train()
        device = next(net.parameters()).device
        start_time = time.time()
        data_time1 = time.time()
        for i, d in enumerate(train_dataloader):

            data_time.update(time.time()-data_time1)
            start_time = time.time()
            d = d.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
   
            if iterations>900000:
                for p in optimizer.param_groups:
                    p['lr'] = 1e-5
            out_net = net(d)
            out_criterion = criterion(out_net, d)
            ortho_loss = net.ortho_loss()
            total_loss = out_criterion["loss"] + args.ortho_weight*ortho_loss
            total_loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()

            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            if i % 50 ==0:
                mse_loss = out_criterion['mse_loss']
                if mse_loss.item() > 0:
                    psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                    psnrs.update(psnr.item())
                else:
                    psnrs.update(100)
                elapsed.update(time.time() - start_time)
                losses.update(out_criterion['loss'].item())
                bpps.update(out_criterion['bpp_loss'].item())
                mse_losses.update(mse_loss.item())
                aux_losses.update(ortho_loss.item())
                
            if iterations % 100 == 0:
             
                current_time = datetime.now()
                print(    ' | '.join([
                f"{current_time}",
                f'Epoch {epoch}',
                f"{i*len(d)}/{len(train_dataloader.dataset)}",
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'DataTime {data_time.val:.3f} ({data_time.avg:.3f})',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'aux_losses Loss {aux_losses.val:.3f} ({aux_losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
            ]))
                start_time = time.time()

            if (iterations% 5000 == 0):
                net.eval()
                loss = test_epoch(iterations, test_dataloader, net, criterion)
                net.train()
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                print("best:",best_loss,"curent:",loss)
                if args.save and iterations% 5000 == 0 :
                    save_checkpoint(
                        { 
                            "epoch": epoch,
                            "state_dict": net.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                        },
                        is_best,
                        args.save_path,
                    )
            iterations = iterations + 1
            data_time1 = time.time()
            if iterations>1000000:
                exit(-1)

 
        
    
    





if __name__ == "__main__":
    main(sys.argv[1:])
