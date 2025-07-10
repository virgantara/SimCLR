import torch
from byol_pytorch import BYOL
from torchvision import models
from tqdm import tqdm
from dataset import UnlabeledImageDataset
from torchvision import transforms
import wandb
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data_path', metavar='DIR', default='../datasets/tiny-imagenet-200/test/images',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='tinyimagenet',
                    help='dataset name', choices=['stl10', 'cifar10','tinyimagenet','isic','isic2024'])
parser.add_argument('-log_dir', default='./logs/simclr',
                    help='log path')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
def main():
    
    
    args = parser.parse_args()
    args.wandb_project = "byol-contrastive-isic"  # your project name
    args.wandb_run_name = "resnet50-aug-v1"    # optional, run identifier
    
    wandb.init(
        project=args.wandb_project,  # pass this via args
        name=args.wandb_run_name,    # optional, also via args
        config=vars(args)
    )
    

    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    learner = BYOL(
        resnet,
        image_size = args.image_size,
        hidden_layer = 'avgpool'
    ).to(device)

    wandb.watch(learner, log='all')

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # def sample_unlabelled_images():
    #     return torch.randn(4, 3, 256, 256)


    train_dataset = UnlabeledImageDataset(
        img_dir=args.data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    n_iter = 0
    for epoch in range(args.epochs):
        for images in tqdm(train_loader):
            images = images.to(device)
            # images = sample_unlabelled_images()
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

            with torch.no_grad():
                z_online = learner.online_encoder(images)[0]  # get projection
                z_target = learner.target_encoder(images)[0]  # get projection
                cos_sim = F.cosine_similarity(z_online, z_target).mean().item()


            wandb.log({
                'loss': loss.item(),
                'cosine_similarity': cos_sim,
                'lr': opt.param_groups[0]['lr'],
                'epoch': epoch,
                'step': n_iter
            })

            n_iter += 1

    # save your improved network
    torch.save(resnet.state_dict(), './byol-improved-net.pt')


if __name__ == "__main__":

    main()
