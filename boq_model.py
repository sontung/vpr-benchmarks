from collections import OrderedDict

import torch
import torchvision.transforms
from PIL import Image
import sys


class BOQModel:
    def __init__(self):
        self.model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288)
        self.model.cuda().eval()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (322, 322),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.conf = {"name": "boq"}

    def process(self, name):
        try:
            image = Image.open(name).convert("RGB")
        except OSError:
            return None
        image = self.transform(image)
        with torch.no_grad():
            image_descriptor = self.model(image.unsqueeze(0).cuda())[0]
            image_descriptor = image_descriptor.squeeze().cpu().numpy()  # 12288
        return image_descriptor


if __name__ == '__main__':
    m = BOQModel()
    m.process("/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/dusk/rear/1424450252005000.jpg")