from pathlib import Path, PurePath
from tqdm import tqdm
import random

scannet_path = 'dataset/scannet_240x320'
# scannet_path = '/data/ScanNet/labels_2d_240x320'

scene_names = [PurePath(x).relative_to(scannet_path) for x in Path(scannet_path).iterdir()]
scene_names = [x for x in scene_names if 'scene' in str(x)]
scene_num = len(scene_names)
print('Found %d names'%scene_num)

for split in ['train', 'val']:
    frame_paths_all_dict = {'color': [], 'label': []}
    if split == 'train':
        scene_names_split = scene_names[:int(scene_num*0.8)]
    else:
        scene_names_split = scene_names[-(scene_num - int(scene_num*0.8)):]
    print('%d names for split %s'%(len(scene_names_split ), split))

    for scene_name in tqdm(scene_names_split):
        scene_path = Path(scannet_path) / scene_name
        frame_paths_dict = {'color': [], 'label': []}
        for subset in ['color', 'label']:
            subset_path = scene_path / subset
            frame_names = [PurePath(x).relative_to(subset_path) for x in Path(subset_path).iterdir()]
            frame_paths = [str((subset_path / frame_name).relative_to(scannet_path)) for frame_name in frame_names]
            frame_paths.sort()
            frame_paths_dict[subset] = frame_paths
        assert len(frame_paths_dict['color']) == len(frame_paths_dict['label'])
        for subset in ['color', 'label']:
            frame_paths_all_dict[subset] += frame_paths_dict[subset]

    for subset in ['color', 'label']:
        random.seed(123456)
        random.shuffle(frame_paths_all_dict[subset])

    print(frame_paths_all_dict['color'][:5], frame_paths_all_dict['label'][:5])

    output_txt_file = Path(scannet_path) / Path('list') / Path('%s.txt'%split)
    with open(str(output_txt_file), 'w') as text_file:
        for path_color, path_label in zip(frame_paths_all_dict['color'], frame_paths_all_dict['label']):
            text_file.write('%s %s\n'%(path_color, path_label))
    