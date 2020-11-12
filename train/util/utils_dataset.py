from collections import OrderedDict
import numpy as np

def get_color_encoding(dataset_name='openrooms'):
    colors = np.loadtxt('data/%s/%s_colors.txt'%(dataset_name, dataset_name)).astype('uint8')
    with open('data/%s/%s_names.txt'%(dataset_name, dataset_name), "r") as f:
        names = f.read().splitlines() 
    assert len(colors) == len(names)
    name_to_color_dict = OrderedDict((name, (color[0], color[1], color[2])) for (name, color) in zip(names, colors))
    return name_to_color_dict

def color_dict_to_text_color_lists(color_dict):
    count_key = 0
    rows = int(np.sqrt(len(color_dict.keys()))) + 1
    cols = len(color_dict.keys()) // rows + 1
    text_list = [[] for _ in range(rows)]
    color_list = [[] for _ in range(rows)]

    for idx, (key, color) in enumerate(color_dict.items()):
    #     print(key, color)
        # if key=='unlabeled':
        #     continue
        text_list[count_key//cols].append('%d-%s'%(idx, key))
        color_list[count_key//cols].append((color[0]/255., color[1]/255., color[2]/255.))
        count_key += 1
    text_list[-1] += ['' for _ in range(rows*cols-count_key)]
    color_list[-1] += [(1, 1, 1) for _ in range(rows*cols-count_key)]

    return text_list, color_list
