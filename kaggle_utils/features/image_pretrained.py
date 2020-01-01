from functools import partial

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base import BaseFeatureTransformer


imagenet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resize_to_square(img):
    old_size = img.shape[:2]
    img_size = max(old_size)
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / (std[i])
    return image


def center_crop(image, size=226):
    image_height, image_width = image.shape[:2]

    if image_height <= image_width and abs(image_width - size) > 1:

        dx = int((image_width - size) / 2)
        image = image[:, dx:-dx, :]
    elif abs(image_height - size) > 1:
        dy = int((image_height - size) / 2)
        image = image[dy:-dy, :, :]

    image_height, image_width = image.shape[:2]
    if image_height is not size or image_width is not size:
        image = cv2.resize(image, (size, size))

    return image


def collate_fn_numpy(batch):
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(targets)
    images = np.concatenate(images, axis=0)
    return [images, targets]


class ImageDatasetFromPath(Dataset):
    def __init__(
        self, paths, labels=None, return_numpy=False,
        preprocessors=[resize_to_square], transforms=None, image_size=224
    ):
        self.paths = paths
        self.image_size = image_size
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros((len(paths)))
        self.return_numpy = return_numpy
        self.preprocessors = preprocessors
        self.transforms = transforms
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        
        if len(self.preprocessors):
            for preprocessor in self.preprocessors:
                image = preprocessor(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if image.max() > 1:
            image = image / 255
            
        if self.return_numpy:
            return (image[np.newaxis, :, :, :], self.labels[index])
            
        # to tensor
        if self.transforms is not None:
            image = self.transforms(Image.fromarray(image))
            if not isinstance(image, dict):
                # for TTA
                if len(image.shape) == 4:
                    return (
                        torch.cat([transforms.ToTensor()(im.squeeze()/255).unsqueeze(0) for im in image]), 
                        torch.tensor(self.labels[index])
                    )
        else:
            image = transforms.ToTensor()(image).type(torch.FloatTensor)
        return (image, torch.tensor(self.labels[index]))        
        

class FasterRCNNFeaturesTransformer(BaseFeatureTransformer):
    def __init__(
        self, path_list, threshold=0.8, device='cuda', preprocessors=[], transforms=None,
        batch_size=16, workers=4, image_size=224, name='',
    ):
        self.path_list = path_list
        self.threshold = threshold
        self.device = device
        self.preprocessors = preprocessors
        self.transforms = transforms
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        self.name = name
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.eval().to(device)

    def transform(self, dataframe):
        dataloader = torch.utils.data.DataLoader(
            ImageDatasetFromPath(
                self.path_list, 
                preprocessors=[resize_to_square]+self.preprocessors, 
                transforms=self.transforms,
                image_size=self.image_size
            ),
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.workers, 
            pin_memory=True
        )
        
        features = []
        with torch.no_grad():
            for i, (input, _) in enumerate(dataloader):
                input = input.to(self.device)
                preds = self.model(input)
                result = []
                for p in preds:
                    score = p['scores'].cpu().detach().numpy()
                    mask = score >= self.threshold
                    n_objects = len(score)
                    n_high_score_objects = sum(mask)
                    if n_high_score_objects:
                        best_box = p['boxes'].cpu().detach().numpy()[0]
                        best_label = p['labels'].cpu().detach().numpy()[0]
                        best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                        best_aspect = (best_box[2] - best_box[0]) / (best_box[3] - best_box[1])
                    else:
                        best_box, best_label, best_area, best_aspect = [0, 0, 0, 0], 0, 0, 0
                    result.append([n_objects, n_high_score_objects, *best_box, best_label, best_area, best_aspect])
                features.append(result)
        features = np.concatenate(features)
        features = pd.DataFrame(
            features, 
            columns=[
                'n_objects', 'n_high_score_objects', 
                'x1', 'y1', 'x2', 'y2', 
                'best_label', 'best_area', 'best_aspect'
            ]
        )
        features.columns = [self.name + c for c in features.columns]
        self.features = [features]
        return pd.concat([dataframe, features], axis=1)


class PytorchPretrainedImageFeaturesTransformer(BaseFeatureTransformer):
    '''
    Pretrained weights can be found here: 
    https://pytorch.org/hub/research-models
    '''
    def __init__(
        self, path_list, device='cuda', preprocessors=[], transforms=imagenet_transforms, 
        version='pytorch/vision:v0.4.2', model_name='resnext50_32x4d',
        batch_size=16, workers=4, image_size=224,
    ):
        self.path_list = path_list
        self.device = device
        self.preprocessors = preprocessors
        self.transforms = transforms
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        self.model_name = model_name
        
        self.model = torch.hub.load(version, model_name, pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.eval().to(device)

    def transform(self, dataframe):
        dataloader = torch.utils.data.DataLoader(
            ImageDatasetFromPath(
                self.path_list, 
                preprocessors=[resize_to_square]+self.preprocessors, 
                transforms=self.transforms,
                image_size=self.image_size
            ),
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.workers, 
            pin_memory=True
        )
        
        features = []
        with torch.no_grad():
            for i, (input, _) in enumerate(dataloader):
                input = input.to(self.device)
                preds = self.model(input).cpu().detach().numpy()
                features.append(preds)
        features = np.concatenate(features)
        features = pd.DataFrame(
            features, 
            columns=[f'{self.model_name}_{i:03}' for i in range(features.shape[1])]
        )
        self.features = [features]
        return pd.concat([dataframe, features], axis=1)


class TFPretrainedImageFeaturesTransformer(BaseFeatureTransformer):
    '''
    Pretrained weights can be found here: 
    https://tfhub.dev/s?module-type=image-feature-vector&q=tf2
    '''
    def __init__(
        self, path_list, 
        classifier_url='https://tfhub.dev/tensorflow/resnet_50/feature_vector/1', 
        preprocessors=[], batch_size=256, workers=4, image_size=224,
    ):
        self.path_list = path_list
        self.classifier_url = classifier_url
        self.preprocessors = preprocessors
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        
        self.model_name = classifier_url.split('/')[-3]
        self.model = tf.keras.Sequential([
            hub.KerasLayer(classifier_url, input_shape=(image_size, image_size, 3,)),
        ])

    def transform(self, dataframe):
        dataloader = torch.utils.data.DataLoader(
            ImageDatasetFromPath(
                self.path_list, 
                return_numpy=True,
                preprocessors=[resize_to_square, partial(center_crop, size=self.image_size), normalize],
                transforms=None,
                image_size=self.image_size,
            ),
            batch_size=self.image_size,
            shuffle=False,
            num_workers=self.workers, 
            collate_fn=collate_fn_numpy,
            pin_memory=False
        )
        
        features = []
        for i, (input, _) in enumerate(dataloader):
            preds = self.model.predict(input)
            features.append(preds)
        features = np.concatenate(features)
        features = pd.DataFrame(
            features, 
            columns=[f'{self.model_name}_{i:03}' for i in range(features.shape[1])]
        )
        self.features = [features]
        return pd.concat([dataframe, features], axis=1)
