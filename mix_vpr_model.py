import sys
import torchvision.transforms as tvf
from PIL import Image

import torch

sys.path.append("../MixVPR")
from mix_vpr_main import VPRModel


def load_image(path):
    try:
        image_pil = Image.open(path).convert("RGB")
    except OSError:
        return None

    # add transforms
    transforms = tvf.Compose([
        tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
        tvf.ToTensor(),
        tvf.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    # apply transforms
    image_tensor = transforms(image_pil)
    return image_tensor


class MVModel:
    def __init__(self, dim=128):
        self.conf = {"name": "mixvpr"}
        if dim == 128:
            self.encoder_global = VPRModel(
                backbone_arch="resnet50",
                layers_to_crop=[4],
                agg_arch="MixVPR",
                agg_config={
                    "in_channels": 1024,
                    "in_h": 20,
                    "in_w": 20,
                    "out_channels": 64,
                    "mix_depth": 4,
                    "mlp_ratio": 1,
                    "out_rows": 2,
                },
            ).cuda()

            state_dict = torch.load(
                "../MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt"
            )
        elif dim == 512:
            self.encoder_global = VPRModel(
                backbone_arch="resnet50",
                layers_to_crop=[4],
                agg_arch="MixVPR",
                agg_config={
                    "in_channels": 1024,
                    "in_h": 20,
                    "in_w": 20,
                    "out_channels": 256,
                    "mix_depth": 4,
                    "mlp_ratio": 1,
                    "out_rows": 2,
                },
            ).cuda()

            state_dict = torch.load(
                "../MixVPR/resnet50_MixVPR_512_channels(256)_rows(2).ckpt"
            )
        else:
            self.encoder_global = VPRModel(
                backbone_arch="resnet50",
                layers_to_crop=[4],
                agg_arch="MixVPR",
                agg_config={
                    "in_channels": 1024,
                    "in_h": 20,
                    "in_w": 20,
                    "out_channels": 1024,
                    "mix_depth": 4,
                    "mlp_ratio": 1,
                    "out_rows": 4,
                },
            ).cuda()

            state_dict = torch.load(
                "../MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
            )
        self.encoder_global.load_state_dict(state_dict)
        self.encoder_global.eval()

    def process(self, name):
        image = load_image(name)
        if image is None:
            return None
        image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
        image_descriptor = image_descriptor.squeeze().cpu().numpy()
        return image_descriptor


if __name__ == "__main__":
    m = MVModel(512)
    m = MVModel(128)
    m = MVModel(256)
