from voc_dataset import VOCDataset
from torchvision.transforms import ToTensor, Compose, RandomAffine, ColorJitter
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch
import argparse
import os
import numpy as np
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Train faster r-cnn model')
    parser.add_argument('--num_epochs', '-n', type=int, default=30)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--data_path', '-d', type=str, default='dataset')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    parser.add_argument('--momentum', '-m', type=float, default=0.9)
    parser.add_argument('--year', '-y', type=str, default='2012')
    parser.add_argument('--log_path', '-p', type=str, default='tensorboard')
    parser.add_argument('--checkpoint_path', '-c', type=str, default='trained_models')
    parser.add_argument('--saved_checkpoint', '-o', type=str, default=None)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images, labels = zip(*batch)
    return images, labels

def train(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = Compose([
        # RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8,1.2), shear=5),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor()
    ])
    train_dataset = VOCDataset(root=args.data_path, year=args.year, image_set='train', download=False, transform=train_transform)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        num_workers=4, 
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = VOCDataset(root=args.data_path, year=args.year, image_set='val', download=False, transform=ToTensor())
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size=args.batch_size,
        num_workers=4, 
        shuffle=False,
        collate_fn=collate_fn
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                                                  trainable_backbone_layers=3)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels,
                                                      num_classes=len(train_dataset.categories))
    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    if args.saved_checkpoint:
        checkpoint = torch.load(
            args.saved_checkpoint,
            map_location=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['map']
    else:
        start_epoch = 0
        best_map = -1

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)

    writer = SummaryWriter(log_dir=args.log_path)
    num_iters_per_epoch = len(train_dataloader)
    for epoch in range(start_epoch, args.num_epochs):
        # TRAIN
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels = [{'boxes': label['boxes'].to(device), 'labels': label['labels'].to(device)} for label in labels]

            # forward
            losses = model(images, labels)
            final_losses = sum(loss for loss in losses.values())

            # backward
            optimizer.zero_grad()
            final_losses.backward()
            optimizer.step()
            train_loss.append(final_losses.item())
            mean_loss = np.mean(train_loss)

            progress_bar.set_description('Epoch {}/{}. Loss {:0.4f}'.format(epoch + 1, args.num_epochs, final_losses.item()))
            writer.add_scalar('Train/Loss', mean_loss, epoch * num_iters_per_epoch + iter)

        # VALIDATION
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="yellow")
        metric = MeanAveragePrecision(iou_type='bbox')
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)
            preds = []
            for output in outputs:
                preds.append({
                    'boxes': output['boxes'].to('cpu'),
                    'scores': output['scores'].to('cpu'),
                    'labels': output['labels'].to('cpu')
                })
            targets = []
            for label in labels:
                targets.append({
                    'boxes': label['boxes'],
                    'labels': label['labels']
                })
            metric.update(preds, targets)

        result = metric.compute()
        pprint(result)
        writer.add_scalar('Val/mAP', result['map'], epoch)
        writer.add_scalar('Val/mAP_50', result['map_50'], epoch)
        writer.add_scalar('Val/mAP_75', result['map_75'], epoch)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'map': result['map'],
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, 'last.pt'))
        if result['map'] > best_map:
            best_map = result['map']
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pt'))


if __name__ == '__main__': 
    args = get_args()
    train(args)
