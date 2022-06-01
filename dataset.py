import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


class ExtractFeatures():
    def __init__(self, cuda=True, model='resnet-18'):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = 512
        self.model_name = model

        # self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
        self.model = models.resnet18(pretrained=True)
        self.layer = model._modules.get('avgpool')
        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(image)
        h.remove()

        return my_embedding.numpy()[0, :, 0, 0]

