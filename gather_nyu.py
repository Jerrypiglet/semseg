from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os
import glob
import os.path as osp


list_path = 'train/data/nyu'
dataset_path = Path('dataset/nyu')

for split in ['train', 'val', 'test']:
# for split in ['train']:
    # scene_num = len(scene_list)
    # if split == 'train':
    #     scene_list_split = scene_list[:int(scene_num*0.95)]
    # elif split == 'val':
    #     scene_list_split = scene_list[-(scene_num - int(scene_num*0.95)):]
    # elif split == 'test':
    #     scene_list_split = scene_listokok
    # # print('%d scenes for split %s'%(len(scene_list_split ), split))

    frame_paths_dict = {'image': [], 'seg40': []}
    for subset in ['image', 'seg40']:
        if split in ['train', 'val']:
            subset_path = dataset_path / subset / 'train'
        else:
            subset_path = dataset_path / subset / 'test'

        phase = 'TRAIN' if split in ['train', 'val'] else 'TEST'
        frame_paths = glob.glob(osp.join(str(dataset_path), subset, phase.lower(),"*.png"))
        frame_paths = sorted(frame_paths)

        random.seed(0)
        random.shuffle(frame_paths)
        frame_paths = [PurePath(x).relative_to(dataset_path) for x in frame_paths]

        frame_num = len(frame_paths)
        train_count = int(frame_num * 0.9)
        val_count = frame_num - train_count

        if split == 'train':
            frame_paths = frame_paths[:-val_count]
            # if subset == 'image':
            #     print([str(Path(x).name) for x in frame_paths])
        elif split == 'val':
            frame_paths = frame_paths[-val_count:]
        # print('=====', subset, split, [Path(x).name for x in frame_paths])

        frame_paths_dict[subset] = frame_paths


    # print(frame_paths_dict['image'][:5], frame_paths_dict['seg40'][:5])

    output_list_path = Path(list_path) / Path('list')
    output_list_path.mkdir(parents=True, exist_ok=True)
    output_txt_file = output_list_path / Path('%s.txt'%split)

    with open(str(output_txt_file), 'w') as text_file:
        for path_cam0, path_label in zip(frame_paths_dict['image'], frame_paths_dict['seg40']):
            text_file.write('%s %s\n'%(path_cam0, path_label))
    print('Wrote to %s'%output_txt_file)
    
# phase = 'TRAIN'
# split = 'train'
# imList = glob.glob(osp.join(str(dataset_path),"image",phase.lower(),"*.png"))
# imList = sorted(imList)
# random.seed(0)
# random.shuffle(imList)
# if phase.upper() == 'TRAIN':
#     num_scenes = len(imList)
#     train_count = int(num_scenes * 0.9)
#     val_count = num_scenes - train_count
#     if split == 'train':
#         imList = imList[:-val_count]
#         print([Path(x).name for x in imList])
#     if split == 'val':
#         imList = imList[-val_count:]