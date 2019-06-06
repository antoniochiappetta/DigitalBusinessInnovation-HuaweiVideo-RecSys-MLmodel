import json
import glob
import errno
import numpy as np
import pprint
import copy

with open('./data_structure.json') as structure:
    file_names = json.load(structure)
    for key in list(file_names.keys()):
        file_names[key] = []

max_values = copy.deepcopy(file_names)

path = './results/ICMtype_ICM_######change######*.json'
files = glob.glob(path)

for file_path1 in files:
    try:
        with open(file_path1) as json_file:
            data1 = json.load(json_file)
            for key1 in list(file_names.keys()):
                file_names[key1].append(file_path1)
                max_values[key1].append(data1[key1])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


for key2 in list(file_names.keys()):
    file_names[key2] = file_names[key2][np.argmax(np.array(max_values[key2]))]
    max_values[key2] = np.max(np.array(max_values[key2]))

# pprint.pprint(file_names)
# pretty_dict_str1 = pprint.pformat(file_names)

# pprint.pprint(max_values)
# pretty_dict_str2 = pprint.pformat(max_values)

with open('./best_results/######change######_tuning_results_file_names.json', 'w') as outfile1:  
    json.dump(file_names, outfile1)
with open('./best_results/######change######_tuning_results_values.json', 'w') as outfile2:  
    json.dump(max_values, outfile2)
