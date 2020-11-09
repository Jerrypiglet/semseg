import os
import os.path
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_root = data_root
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        if 'InteriorNet' in self.data_root:
            label = np.array(Image.open(label_path).convert('L'))
        else:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # print(np.amax(label), np.amin(label), np.median(label), label.shape, label.dtype, label_path)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
            # print(torch.amax(label), torch.amin(label), self.data_root, '-------')
            if 'scannet' in self.data_root:
                label = nyu40_to_scannet20(label)
                label[label > 20] = 0
        return image, label, image_path

def nyu40_to_scannet20(label):
	"""Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

	# Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
	# Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in 
	# 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
	# http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
	# http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
	# The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

	# The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the 
	# corresponding target 'scannet20' label
	remapping = [(0,0),(13,0),(15,0),(17,0),(18,0),(19,0),(20,0),(21,0),(22,0),(23,0),(25,0),(26,0),(27,0),
				(29,0),(30,0),(31,0),(32,0),(35,0),(37,0),(38,0),(40,0),(14,13),(16,14),(24,15),(28,16),(33,17),
				(34,18),(36,19),(39,20)]
	for src, tar in remapping:
		label[label==src] = tar
	return label

