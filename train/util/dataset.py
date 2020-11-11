import os
import os.path
import cv2
import numpy as np
from PIL import Image
from pathlib import Path, PurePath

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
    # print(image_label_list[:5])
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, logger=None, is_master=False, args=None):
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.data_root = data_root
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.logger = logger
        self.is_master = is_master
        self.args = args
        self.dataset_name = args.dataset_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        image_path, label_path = self.data_list[index]
        image = self.read_image(image_path)
        # print(image_path)
        label = self.read_label(label_path)
        # print(np.amax(label), np.amin(label), np.median(label), label.shape, label.dtype, label_path)

        if image is None:
            raise (RuntimeError("Image is NONE: " + image_path + "\n"))
        # print(image.shape)
        if label is None:
            # print(list(Path(label_path).parent.iterdir()))
            raise (RuntimeError("Label is NONE: " + label_path + "\n"))
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform is not None:
            image, label = self.transform(image, label)
            # print(torch.amax(label), torch.amin(label), self.data_root, '-------')
            if 'scannet' in self.dataset_name.lower():
                label = nyu40_to_scannet20(label)
                label[label > 20] = 0
        return image, label, image_path

    def read_image(self, image_path):
        # if self.is_master:
        # print('======'+image_path)
        if 'openrooms' in self.dataset_name.lower():
            im_hdr = self.loadHdr(image_path)
            seg = np.ones((1, im_hdr.shape[1], im_hdr.shape[2]))
            im_hdr, scale = self.scaleHdr(im_hdr, seg)
            im_not_hdr = np.clip(im_hdr**(1.0/2.2), 0., 1.)
            image = (255. * im_not_hdr).transpose(1, 2, 0).astype(np.uint8)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            # print(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image)
        return image

    def read_label(self, label_path):
        if 'interiornet' in self.dataset_name.lower():
            label = np.array(Image.open(label_path).convert('L'))
        elif 'openrooms' in self.dataset_name.lower():
            label = np.load(label_path)
            # if np.amax(label) > 42:
            #     print(np.amax(label), np.amin(label))
            label += 1 # to make 0 as unlabelled, 1 as environment
        else:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        return label


    def loadHdr(self, imName):
        if not(os.path.isfile(imName ) ):
            print(imName )
            assert(False )
        im = cv2.imread(imName, -1)
        # print(imName, im.shape, im.dtype)

        if im is None:
            print(imName )
            assert(False )
        # im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg):
        # print(hdr.shape) # (3, 480, 640)
        imWidth, imHeight = hdr.shape[2], hdr.shape[1]
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split == 'train':
            scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * imWidth * imHeight * 3) ], 0.1, None)
            print('scaling...')
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * imWidth * imHeight * 3) ], 0.1, None)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 


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

