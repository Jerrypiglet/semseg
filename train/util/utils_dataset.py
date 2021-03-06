from collections import OrderedDict
import numpy as np

nyu_or_dict = {43:1, 44:2, 28:3, 42:3, 18:4, 21:5, 11:6, 4:7, 41:8, 31:9, 10:10, 7:12, 5:14, 1:16, 16:18, 45:22, 24:30, 32:35, 25:36, 37:37, 8:38, 26:38, \
    3:39, 6:39, 13:39, 14:39, 40:39, \
    9:40, 12:40, 15:40, 17:40, 19:40, 20:40, 22:40, 23:40, 27:40, 30:40, 33:40, 34:40, 35:40, 36:40, 38:40, 39:40, 2:0, 29:0}

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

def map_to_nyu(input_array, dataset_name):
    if dataset_name == 'openrooms':
        input_array = map_openrooms_nyu(input_array)
        print('mapping from OR')
    elif dataset_name == 'InteriorNet':
        input_array = map_InteriorNet_nyu(input_array)
        print('mapping from INet')
    elif dataset_name == 'nyu':
        return input_array
    else:
        raise (RuntimeError('No support to convert from %d to NYU label space!'%dataset_name))
    return input_array


def map_openrooms_nyu(input_array):
    # keys = list(nyu_or_dict.keys())
    # print(keys)
    nyu_or_map = lambda x: nyu_or_dict.get(x, x)
    nyu_or_map = np.vectorize(nyu_or_map)
    return nyu_or_map(input_array)

def map_openrooms_nyu_gpu(input_tensor):
    nyu_or_map_gpu = lambda x: nyu_or_dict.get(x, x)
    return input_tensor.apply_(nyu_or_map_gpu)

def map_InteriorNet_nyu(input_array):
    # keys = list(nyu_or_dict.keys())
    # print(keys)
    # nyu_or_map = lambda x: nyu_or_dict.get(x, x)
    # nyu_or_map = np.vectorize(nyu_or_map)
    # return nyu_or_map(input_array)
    return input_array

def map_InteriorNet_nyu_gpu(input_tensor):
    # keys = list(nyu_or_dict.keys())
    # print(keys)
    # nyu_or_map = lambda x: nyu_or_dict.get(x, x)
    # nyu_or_map = np.vectorize(nyu_or_map)
    # return nyu_or_map(input_tensor)
    return input_tensor