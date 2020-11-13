from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os

list_path = Path('train/data/openrooms/list')

subsample_ratio_name_dict = {'1': '100k', '0.5': '50k', '0.3': '30k', '0.1': '10k'}

for split in ['train']:
    trainFile = list_path / (split + ".txt")
    with open(trainFile, 'r') as fIn:
        trainList = fIn.readlines()
    trainList = [x.strip() for x in trainList]
    trainList = sorted(trainList)
    random.seed(0)
    random.shuffle(trainList)
    num_scenes = len(trainList)

    for subsample_ratio in subsample_ratio_name_dict.keys():
        subsample_ratio_float = float(subsample_ratio)
        subset_size = int(subsample_ratio_float * num_scenes)
        trainList = trainList[:subset_size]

        output_txt_file = list_path / Path('%s.txt'%split)
        output_txt_file = str(output_txt_file).replace('.txt', '_%s.txt'%subsample_ratio_name_dict[subsample_ratio])

        with open(str(output_txt_file), 'w') as text_file:
            for x in trainList:
                text_file.write('%s\n'%(x))
        print('Wrote to %s'%output_txt_file)