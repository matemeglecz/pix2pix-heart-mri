from data.base_dataset import BaseDataset, get_transform, get_params
import data.mapping_utils as mapping_utils
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import random
import torch

TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = []#[3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]
TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]
INTEROBSERVER_PATIENTS = range(223, 262+1)

class SeDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):        
        parser.add_argument('--split', type=str, default="train", help="train, val, test, interobserver")
        parser.add_argument('--observer_id', type=int, default=1, help="1, 2, 3")
        parser.add_argument('--mapping_only', action='store_true', help="Only use mapping samples")
        
        
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(no_flip=True)

        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt

        self.root = opt.dataroot

        self.transforms = None#transforms
        self.split = opt.split

        if opt.split == "interobserver" and opt.observer_id != 1:
            contours_filename = f"Contours_{opt.observer_id}.json"
        else:
            contours_filename = f"Contours.json"
        self.all_samples = mapping_utils.construct_samples_list(
            self.root, contours_filename
        )
        mapping_utils.print_diagnostics(self.root, self.all_samples)

        segmentation_partitions = {
            "train": TRAIN_PATIENTS,
            "val": VAL_PATIENTS,
            "test": TEST_PATIENTS,
            "interobserver": INTEROBSERVER_PATIENTS,
        }

        self.samples = mapping_utils.split_samples_list(
            self.all_samples, segmentation_partitions[self.split]
        )
        #if opt.mapping_only:
            #self.to_mapping_only()

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict: (sample, target)
        """
        path, target_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None)
        target_contours = mapping_utils.load_contours(target_path)
        target = mapping_utils.contours_to_masks(target_contours, sample.shape)
        if self.transforms is not None:
            if "bboxes" in self.transforms.processors.keys():
                bbox = self.compute_bbox(target)
                transformed = self.transforms(image=sample, mask=target, bboxes=[bbox])
            else:
                transformed = self.transforms(image=sample, mask=target)
            sample, target = transformed["image"], transformed["mask"]
        
        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        sample = resize(sample, (256, 256), anti_aliasing=True)
        
        target = resize(target, (256, 256), anti_aliasing=False, mode='edge', preserve_range=True, order=0)

        sample = np.expand_dims(sample, 0)
        target = np.expand_dims(target, 0)

        sample = sample.astype(np.float32)
        target = target.astype(np.float32)

        if self.opt.preprocess != 'none' and random.random() < 0.33:
            # apply the same transform to both A and B            
            transform_params = get_params(self.opt, sample.shape[1:])
            B_transform = get_transform(self.opt, transform_params, method=transforms.InterpolationMode.NEAREST, grayscale=(self.opt.input_nc == 1), convert=False)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1), convert=False)
                      
            sample = torch.from_numpy(sample)
            target = torch.from_numpy(target)
            
            sample = A_transform(sample)
            target = B_transform(target)

            sample = sample.numpy()
            target = target.numpy()
        

        input_dict = {'B': target,
                      'A': sample,
                      'B_paths': target_path,
                      'A_paths': path,
                      }

        return input_dict

    def compute_bbox(self, mask):
        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox = [
            mask.nonzero()[1].min(),  # / mask.shape[0],
            mask.nonzero()[0].min(),  # / mask.shape[1],
            mask.nonzero()[1].max(),  # / mask.shape[0],
            mask.nonzero()[0].max(),  # / mask.shape[1],
            "dummy_label",
        ]  # x_min, y_min, x_max, y_max
        return bbox

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]


    def __crop(img, pos, size):
        ow, oh = img.shape[1:]
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img