import sys

import torchvision
from PIL import Image

sys.path.append("../Patch-NetVLAD")
import configparser
import os
import torch
from os.path import join, isfile

from patchnetvlad.models.models_generic import get_backend as get_backend_patchnetvlad
from patchnetvlad.models.models_generic import get_model as get_model_patchnetvlad
from patchnetvlad.models.models_generic import get_pca_encoding as get_pca_encoding_patchnetvlad
from download_models import download_all_models as download_patchnetvlad



class PatchNetVladModel:
    def __init__(self):
        self.conf = {"name": "patchnetvlad"}

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((480, 640)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        patchnetvlad_root_dir = "../Patch-NetVLAD/patchnetvlad"

        configfile = f"{patchnetvlad_root_dir}/configs/performance.ini"
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)

        device = torch.device("cuda")
        encoder_dim, encoder = get_backend_patchnetvlad()

        if config['global_params']['num_pcs'] != '0':
            resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'
        else:
            resume_ckpt = config['global_params']['resumePath'] + '.pth.tar'
        resume_ckpt = resume_ckpt.split("./")[-1]

        # backup: try whether resume_ckpt is relative to PATCHNETVLAD_ROOT_DIR
        if not isfile(resume_ckpt):
            resume_ckpt = join(patchnetvlad_root_dir, resume_ckpt)
            if not isfile(resume_ckpt):
                download_patchnetvlad(ask_for_permission=False)

        self.model = None
        self.config = config
        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            if config['global_params']['num_pcs'] != '0':
                assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            if config['global_params']['num_pcs'] != '0':
                use_pca = True
            else:
                use_pca = False
            model = get_model_patchnetvlad(encoder, encoder_dim, config['global_params'], append_pca_layer=use_pca)
            model.load_state_dict(checkpoint['state_dict'])

            model = model.to(device)
            model.eval()
            self.model = model
            print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image).cuda()
        
        with torch.no_grad():
            image_encoding = self.model.encoder(image.unsqueeze(0))
            vlad_local, vlad_global = self.model.pool(image_encoding)

            image_descriptor = get_pca_encoding_patchnetvlad(self.model, vlad_global)
        image_descriptor = image_descriptor.squeeze().cpu().numpy()  # 4096
        return image_descriptor


if __name__ == "__main__":
    du = PatchNetVladModel()
    g = du.process(
        "/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/dawn/rear/1418721355257224.jpg"
    )
    print(g.shape)
    print()
