import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import random
import sys
import traceback
import logging
import igl
import numpy as np
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def to_off(path, task_name, no_scale=False, bbox_spec=None):

    file_path = os.path.dirname(path)
    # data_type = file_path.split('/')[2]
    #print(data_type)
    file_name = os.path.splitext(os.path.basename(path))[0]
    output_file = os.path.join(file_path, file_name + '_scaled.off')

    print("preprocessing data in {}".format(file_path))
    
    if os.path.exists(output_file):
        print('Exists: {}'.format(output_file))
        print('Remove: {}'.format(output_file))
        os.remove(output_file)

    try:

        v, f = igl.read_triangle_mesh(path)
        if bbox_spec is None:
            bb_max = v.max(axis=0, keepdims=True)
            bb_min = v.min(axis=0, keepdims=True)
        else:
            bb_max = bbox_spec[1]
            bb_min = bbox_spec[0]
        ori_bbox = np.concatenate((bb_min, bb_max), axis=0)
        #print(centers)
        #print(bb_max-bb_min)
        if task_name == 'c3d':
            v/=40
        elif task_name != 'arm' and task_name != 'cabinet':
            centers = (bb_max+bb_min)/2.0
            v = v-centers
            if not no_scale:
                v = v/(bb_max-bb_min)
    
        igl.write_triangle_mesh(output_file, v, f) 

        bb_max = v.max(axis=0, keepdims=True)
        bb_min = v.min(axis=0, keepdims=True)
        bbox = np.concatenate((bb_min, bb_max), axis=0)
        print('Finished: {}'.format(path))
        return bbox, ori_bbox
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

