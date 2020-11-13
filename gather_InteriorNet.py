from pathlib import Path, PurePath
from tqdm import tqdm
import random

# dataset_path = 'dataset/InteriorNet_mini'
dataset_path = 'dataset/InteriorNet'
list_path = 'train/data/InteriorNet'

scene_names = [PurePath(x).relative_to(dataset_path) for x in Path(dataset_path).iterdir()]
scene_names = [x for x in scene_names if ('.zip' not in str(x) and 'list' not in str(x))]
scene_num = len(scene_names)
print('Found %d names'%scene_num, scene_names[:5])

subsample_ratio = 0.25
subsample_ratio_name_dict = {'1': '400k', '0.25': '100k', '0.125': '50k', '0.083': '30k', '0.025': '10k'}
assert str(subsample_ratio) in list(subsample_ratio_name_dict.keys())

for split in ['train', 'val', 'test']:
    frame_paths_all_dict = {'cam0/data': [], 'label0/data': []}
    train_val_split_at = int(scene_num * 0.93)
    val_test_split_at = int(scene_num * 0.95)
    if split == 'train':
        # scene_names_split = scene_names[:int(scene_num*0.98)]
        scene_names_split = scene_names[:train_val_split_at]
    elif split == 'val':
        scene_names_split = scene_names[train_val_split_at:val_test_split_at]
        # scene_names_split = scene_names[-(scene_num - int(scene_num*0.98)):]
    elif split == 'test':
        scene_names_split = scene_names[val_test_split_at:]
    print('%d names for split %s'%(len(scene_names_split ), split))

    for scene_name in tqdm(scene_names_split):
        scene_path = Path(dataset_path) / scene_name
        frame_paths_dict = {'cam0/data': [], 'label0/data': []}
        for subset in ['cam0/data', 'label0/data']:
            subset_path = scene_path / subset
            frame_names = [PurePath(x).relative_to(subset_path) for x in Path(subset_path).iterdir()]
            frame_paths = [str((subset_path / frame_name).relative_to(dataset_path)) for frame_name in frame_names]
            if subset == 'label0/data':
                frame_paths = [x for x in frame_paths if '_nyu.png' in x]
                frame_names = [str(x).split('_')[0] for x in frame_names if '_nyu.png' in str(x)]
            # frame_paths.sort()
            frame_paths = [x for _,x in sorted(zip(frame_names, frame_paths))]
            # print(frame_names)
            frame_paths_dict[subset] = frame_paths
        # print(frame_paths_dict['cam0/data'])
        # print(frame_paths_dict['label0/data'])

        # print(frame_paths_dict['cam0/data'])
        # print(frame_paths_dict['label0/data'])
        assert len(frame_paths_dict['cam0/data']) == len(frame_paths_dict['label0/data']), '%d != %d'%(len(frame_paths_dict['cam0/data']), len(frame_paths_dict['label0/data']))
        for subset in ['cam0/data', 'label0/data']:
            frame_paths_all_dict[subset] += frame_paths_dict[subset]

    for subset in ['cam0/data', 'label0/data']:
        random.seed(123456)
        random.shuffle(frame_paths_all_dict[subset])

    print(frame_paths_all_dict['cam0/data'][:5], frame_paths_all_dict['label0/data'][:5])

    if split == 'train':
        for subsample_ratio in list(subsample_ratio_name_dict.keys()):
            subsample_ratio_float = float(subsample_ratio)
            if subsample_ratio_float != 1.:
                index_list = range(len(frame_paths_all_dict['cam0/data']))
                sample_num = int(len(index_list) * subsample_ratio_float)
                print(sample_num, len(index_list), subsample_ratio_float)
                index_list_sample = random.sample(index_list, sample_num)
                im_list = [frame_paths_all_dict['cam0/data'][i] for i in index_list_sample]
                label_list = [frame_paths_all_dict['label0/data'][i] for i in index_list_sample]
            else:
                im_list = frame_paths_all_dict['cam0/data']
                label_list = frame_paths_all_dict['label0/data']

            output_txt_file = Path(list_path) / Path('list') / Path('%s.txt'%split)
            output_txt_file = str(output_txt_file).replace('.txt', '_%s.txt'%(subsample_ratio_name_dict[subsample_ratio]))

            with open(str(output_txt_file), 'w') as text_file:
                for path_cam0, path_label in zip(im_list, label_list):
                    text_file.write('%s %s\n'%(path_cam0, path_label))
    else:
        output_txt_file = Path(list_path) / Path('list') / Path('%s.txt'%split)

        with open(str(output_txt_file), 'w') as text_file:
            for path_cam0, path_label in zip(im_list, label_list):
                text_file.write('%s %s\n'%(path_cam0, path_label))
    