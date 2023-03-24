from operator import indexOf
import os
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import data.mapping_utils as mapping_utils
#import pandas as pd
import glob
#from tabulate import tabulate
from tqdm import tqdm
#import utils
from skimage.transform import resize

DATASET_MEAN = 243.0248
DATASET_STD = 391.5486

# Test and validation samples are selected on a patient level (i.e. all data of a patient is either training, test, or validation)
# Test and validation patients were selected randomly
# from the list of patients without missing contours, or incorrect mapping files
# fmt: off
# fmt: off prevents black from autoformatting these lines
TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = [3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]
TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]
INTEROBSERVER_PATIENTS = range(223, 262+1)

# Adding HCM patients
HCM_TRAIN = [282, 326, 359, 327, 279, 302, 285, 308, 301, 333, 361, 269, 317, 305, 311, 266, 272, 312, 342, 340, 334, 339, 357, 356, 335, 264, 324, 293, 291, 306, 347, 341, 363, 268, 292, 319, 332, 349, 267, 360, 320, 343, 358, 309, 330, 278, 284, 300, 350, 277, 316, 345, 299, 313, 352, 353, 295, 263, 276, 294, 270, 281, 325, 364, 321, 315, 273, 288, 362, 314, 310, 329, 355, 322, 337, 338, 303, 351, 296, 283, 323, 336]
HCM_VAL = [280, 286, 344, 287, 298, 328, 354, 348, 304, 365]
HCM_TEST = [274, 271, 289, 265, 307, 318, 297, 346, 275, 331, 290]

ATHLETE_TRAIN = [366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 380, 382, 383, 384, 385, 386, 387, 388, 389, 392, 393, 394, 395, 396, 397, 398, 400, 401, 402, 404, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 424, 427, 428, 429, 430, 431, 433, 435, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448]
ATHLETE_VAL = [373, 374, 379, 390, 399, 425, 426, 438]
ATHLETE_TEST = [378, 381, 391, 403, 405, 410, 421, 432, 434]

AMYLOIDOSIS_TRAIN = [452, 454, 456, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 470, 471, 472, 473, 475, 476, 477, 478]
AMYLOIDOSIS_VAL = [453, 455, 457, 469, 474, 479, 480]

AKUT_MYOCARDITIS_TRAIN = [
    1262, 1264, 1265, 1266, 1268, 1270, 1271, 1272, 1274, 1275, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1303, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1441, 1487, 1488, 1489, 1491, 1492, 1494, 1496, 1498, 1499, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1534, 1535, 1536, 1537, 1538, 1540, 1541, 1542, 1546, 1554, 1555, 1556] # num = 96
AKUT_MYOCARDITIS_VAL = [
    1267, 1277, 1304, 1445, 1490, 1493, 1497, 1500, 1519, 1533, 1539, 1544] # num = 12
AKUT_MYOCARDITIS_TEST = [
    1263, 1269, 1273, 1276, 1299, 1302, 1442, 1495, 1518, 1543, 1545, 1553] # num = 12

LEZAJLOTT_MYOCARDITIS_TRAIN = [
    1315, 1318, 1319, 1322, 1323, 1326, 1327, 1329, 1331, 1332, 1333, 1335, 1336, 1337, 1338, 1440, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1461, 1463, 1464, 1465, 1466, 1467, 1468, 1470, 1471, 1472, 1474, 1477, 1478, 1482, 1483, 1484, 1485, 1486, 1548, 1550, 1551, 1552] # num = 52
LEZAJLOTT_MYOCARDITIS_VAL = [
    1316, 1317, 1320, 1321, 1334, 1339, 1459, 1469, 1475, 1480, 1481] # num = 11
LEZAJLOTT_MYOCARDITIS_TEST = [
    1324, 1325, 1328, 1330, 1446, 1460, 1462, 1473, 1476, 1479, 1549] # num = 11

# WHEN ADDING NEW PATIENTS, DON'T FORGET TO ADD THEM TO THE APROPRIATE DATASET CLASSES! (or improve this coding scheme)
#fmt: off


class MappingDatasetAlbu(Dataset):
    """ Pytorch dataset for the SE dataset that supports Albumentations augmentations (including bounding box safe cropping).

		Args:
			root (str, Path): Root directory of the dataset containing the 'Patient (...)' folders.
			transforms (albumentations.Compose, optional): Albumentations augmentation pipeline. Defaults to None.
			split (str, optional):  The dataset split, supports `train`, `val`, `test` or `interobserver`. Defaults to 'train'.
			check_dataset (bool, optional): Check dataset for missing and unexpected files. Outdated... Defaults to False.
			observer_id (int, optional): Contours from 3 annotators are available for the same patient in the `interobserver` `split`. `observer_id` selects from these. Supports 1,2 or 3. Defaults to 1.
			mapping_only (bool, optional): Include T1 and T2 mappping images only or also include the map sequences that were used to construct these mappings. Defaults to False.
	"""
    def __init__(self, 
                root, 
                transforms=None, 
                split='train', 
                check_dataset=False, 
                observer_id = 1,
                mapping_only = False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        if check_dataset:
            mapping_utils.check_datset(self.root)
        if split == "interobserver" and observer_id != 1:
            contours_filename = f"Contours_{observer_id}.json"
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
        if mapping_only:
            self.to_mapping_only()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
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
        sample = np.expand_dims(sample, 0)
        target = resize(target, (256, 256), anti_aliasing=False, mode='edge', preserve_range=True, order=0)

        target = np.expand_dims(target.astype(np.float32), axis=0)
        
        return sample, target

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


class MappingUnlabeledDatasetAlbu(Dataset):

    def __init__(self, 
                 root, 
                 transforms=None, 
                 split='train', 
                 mapping_only = False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split

        self.all_samples = mapping_utils.consturct_unlabelled_samples_list(self.root)
        mapping_utils.print_diagnostics(self.root, self.all_samples)
        
        
        UNLABELED = list(range(481, 1262)) # patients without any label (neither segmentation nor diagnosis)
        unlabeled_partitions = {
            "train": TRAIN_PATIENTS + HCM_TRAIN + ATHLETE_TRAIN + AMYLOIDOSIS_TRAIN + AKUT_MYOCARDITIS_TRAIN + LEZAJLOTT_MYOCARDITIS_TRAIN + UNLABELED,
            "val": VAL_PATIENTS + HCM_VAL + ATHLETE_VAL + AMYLOIDOSIS_VAL + AKUT_MYOCARDITIS_VAL + LEZAJLOTT_MYOCARDITIS_VAL,
            "test": TEST_PATIENTS + HCM_TEST + ATHLETE_TEST + AKUT_MYOCARDITIS_TEST + LEZAJLOTT_MYOCARDITIS_TEST,
        }

        self.samples = mapping_utils.split_samples_list(
            self.all_samples, unlabeled_partitions[self.split]
        )
        if mapping_only:
            self.to_mapping_only()
        print("Unlabeled dataset size:", len(self), "images")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None)
        if self.transforms is not None:
            sample = self.transforms(sample)
       
        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        if isinstance(sample, list):
            sample = [np.expand_dims(s['image'], 0) for s in sample]
        else:
            sample = np.expand_dims(sample['image'], 0)

        return sample, 0

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [x for x in self.samples if "_Mapping_" in x]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            # A.LongestMaxSize(192),
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.Normalize(mean=(243.0248,), std=(391.5486,)),
            # ToTensorV2()
        ]
    )

    train_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping",
        transform,
        check_dataset=False,
        mapping_only=True,
    )
    print("Train samples =", len(train_ds))
    val_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="val", mapping_only=True
    )
    print("Val samples =", len(val_ds))
    test_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="test", mapping_only=True
    )
    print("Test samples =", len(test_ds))
    interobs_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="interobserver", mapping_only=True
    )

    # from torch.utils.data import DataLoader

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # print("Train dataloader = ", len(train_loader))

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # print("Val dataloader = ", len(val_loader))

    # train_ds.to_mapping_only()
    print("Train samples =", len(train_ds))
    print("Val samples =", len(val_ds))
    print("Test samples =", len(test_ds))
    print("Interobserver samples =", len(interobs_ds))
    # print("Train dataloader = ", len(train_loader))
    # print("Val dataloader = ", len(val_loader))

    # img, mask = train_ds[10]
    # print("Img shape", img.shape)
    # print("Mask shape", mask.shape)
    # img = img[0,...]#.detach().numpy()
    # cv2.imwrite(f"example_transformed.png", np.concatenate([img, np.ones((img.shape[0], 3))*255, mask], axis=1))
