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

def map_openrooms_nyu(input_array):
    nyu_or_dict = {43:1, 44:2, 28:3, 42:3, 18:4, 21:5, 11:6, 4:7, 41:8, 31:9, 10:10, 7:12, 5:14, 1:16, 16:18, 45:22, 24:30, 32:35, 25:36, 37:37, 8:38, 26:38, \
            3:39, 6:39, 13:39, 14:39, 40:39, \
            9:40, 12:40, 15:40, 17:40, 19:40, 20:40, 22:40, 23:40, 27:40, 30:40, 33:40, 34:40, 35:40, 36:40, 38:40, 39:40, 2:0, 29:0}
        # 0:255, 1:40, 2:41, 3:24, 4:15, 5:18, 6:8, 7:4, 8:38, 9:27,
        # 10:7, 11:255, 12:4, 13:255, 14:4, 15:7, 16:1, 17: 255, 18:13, 19:255,
        # 20: 255, 21:255, 22:42, 23: 255, 24: 255, 25:20, 26: 255, 27: 18, 28: 255, 29: 255,
        # 30:255, 31:255, 32:255, 33:255, 34:32, 35:28, 36: 21, 37:33, 38:5, 39:3, 40:6}
    keys = list(nyu_or_dict.keys())
    # print(keys)
    nyu_or_map = lambda x: nyu_or_dict.get(x, x)
    nyu_or_map = np.vectorize(nyu_or_map)
    return nyu_or_map(input_array)