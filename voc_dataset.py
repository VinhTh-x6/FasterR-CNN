from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
import torch

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index):
        image, data = super().__getitem__(index)
        target = {}
        all_boxes = []
        all_labels = []
        for obj in data['annotation']['object']:
            xmax = int(obj['bndbox']['xmax'])
            xmin = int(obj['bndbox']['xmin'])
            ymax = int(obj['bndbox']['ymax'])
            ymin = int(obj['bndbox']['ymin'])
            all_boxes.append([xmin, ymin, xmax, ymax])
            all_labels.append(self.categories.index(obj['name']))
        all_boxes = torch.FloatTensor(all_boxes)
        all_labels = torch.LongTensor(all_labels)
        target = {
            'boxes': all_boxes,
            'labels': all_labels
        }
        return image, target

if __name__ == '__main__':
    transform = ToTensor()
    train_dataset = VOCDataset(root='dataset', year='2012', image_set='train', download=False, transform=transform)
    data = train_dataset[2000]
    print(data)
    