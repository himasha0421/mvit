# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import json
import os
import random
import re
import pandas as pd
import mvit.utils.logging as logging
import torch
import torch.utils.data
from mvit.utils.env import pathmgr
from PIL import Image
from torchvision import transforms as transforms_tv

from .build import DATASET_REGISTRY
from .transform import transforms_imagenet_train

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Charnet(torch.utils.data.Dataset):
    """CharNet dataset."""

    def __init__(self, cfg, mode, num_retries=10):
        
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for CharNet".format(mode)
        logger.info("Constructing Charnet Dataset {}...".format(mode))
        
        self._construct_chardb()

    def _construct_chardb(self):
        """Constructs the chardb."""
        
        #read the split csv file
        split_path = os.path.join(self.data_path , f'{self.mode}_dataset.csv' )
        logger.info("{} csv data path: {}".format(self.mode, split_path))
        
        # read the csv file
        df_ = pd.read_csv( split_path )
        n_classes = len(df_.label.value_counts())
        # Construct the image db
        self._chardb = []
        
        for img_path , i_label in zip( df_.img_path.values , df_.label.values  ):
            im_path = os.path.join( "dataset" , img_path )
            self._chardb.append({"im_path": im_path, "class":  i_label })
                
        logger.info("Number of images: {}".format(len(self._chardb)))
        logger.info("Number of classes: {}".format( n_classes ))

    def _prepare_im(self, im_path):
        # read the image and coneert to rgb
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )

        if self.mode == "train":
            aug_transform = transforms_imagenet_train(
                img_size=(train_size, train_size),
                color_jitter=self.cfg.AUG.COLOR_JITTER,
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
                re_prob=self.cfg.AUG.RE_PROB,
                re_mode=self.cfg.AUG.RE_MODE,
                re_count=self.cfg.AUG.RE_COUNT,
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
            )
        else:
            t = []
            if self.cfg.DATA.VAL_CROP_RATIO == 0.0:
                t.append(
                    transforms_tv.Resize((test_size, test_size), interpolation=3),
                )
            else:
                # size = int((256 / 224) * test_size) # = 1/0.875 * test_size
                size = int((1.0 / self.cfg.DATA.VAL_CROP_RATIO) * test_size)
                t.append(
                    transforms_tv.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms_tv.CenterCrop(test_size))
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
            aug_transform = transforms_tv.Compose(t)
        im = aug_transform(im)
        return im

    def __load__(self, index):
        try:
            # Load the image
            im_path = self._chardb[index]["im_path"]
            # Prepare the image for training / testing
            if self.mode == "train" and self.cfg.AUG.NUM_SAMPLE > 1:
                im = []
                for _ in range(self.cfg.AUG.NUM_SAMPLE):
                    crop = self._prepare_im(im_path)
                    im.append(crop)
                return im
            else:
                im = self._prepare_im(im_path)
                return im

        except Exception as e:
            print(e)
            return None

    def __getitem__(self, index):
        # if the current image is corrupted, load a different image.
        for _ in range(self.num_retries):
            im = self.__load__(index)
            # Data corrupted, retry with a different image.
            if im is None:
                index = random.randint(0, len(self._chardb) - 1)
            else:
                break
        # Retrieve the label
        label = self._chardb[index]["class"]
        if isinstance(im, list):
            label = [label for _ in range(len(im))]
        # one-hot encode the labels
        label = torch.nn.functional.one_hot( torch.tensor( label ) , num_classes=59 )
        
        return im, label

    def __len__(self):
        return len(self._chardb)
