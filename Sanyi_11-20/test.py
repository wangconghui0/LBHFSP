
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from FFSPTester import FFSPTester as Tester


##########################################################################################
# parameters

score = []
time_str = []

Job_list = [42, 60, 50, 120, 104, 114, 156, 168, 200, 188]

for i in range(8,9):
    env_params = {
        'stage_cnt': 34,
        'machine_cnt_list': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 3, 1, 1, 1, 1, 4, 3, 1, 1, 1, 3, 1, 5, 1, 5, 1, 5, 1],
        'job_cnt': Job_list[i],
        'process_time_params': {
            'time_low': 1,
            'time_high': 10,
        },
        'pomo_size': 720,  # assuming 4 machines at each stage! 4*3*2*1
    }

    model_params = {
        'job_cnt': env_params['job_cnt'],
        'stage_cnt': env_params['stage_cnt'],
        'machine_cnt_list': env_params['machine_cnt_list'],
        'embedding_dim': 256,
        'sqrt_embedding_dim': 256 ** (1 / 2),
        'encoder_layer_num': 3,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16 ** (1 / 2),
        'head_num': 16,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1 / 2) ** (1 / 2),
        'ms_layer2_init': (1 / 16) ** (1 / 2),
        'eval_type': 'softmax',
        'one_hot_seed_cnt': 20,  # must be >= machine_cnt
    }

    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            'path': './result/20240702_123636_matnet_train50_machine[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1]',

            # directory path of pre-trained model and log files saved.
            'epoch': 100,  # epoch version of pre-trained model to load.
        },
        # 'saved_problem_folder': "data/" +str(env_params['job_cnt'])+'/'+ str(env_params['machine_cnt_list']) + '/',
        'saved_problem_folder': "data/",
        'saved_problem_filename': str(i) + '.pt',
        'problem_count': 1,
        'test_batch_size':1,
        'augmentation_enable': True,
        'aug_factor': 1,
        'aug_batch_size': 300,
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    logger_params = {
        'log_file': {
            'desc': 'ffsp_matnet_test_job'+str(env_params['job_cnt'])+'_aug_'+str(tester_params['aug_factor']),
            'filename': 'log.txt'
        }
    }


    ##########################################################################################
    # main

    def main():

        if DEBUG_MODE:
            _set_debug_mode()

        create_logger(**logger_params)
        _print_config()

        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)

        copy_all_src(tester.result_folder)

        aug_score, elapsed_time_str = tester.run()

        return aug_score, elapsed_time_str


    def _set_debug_mode():
        tester_params['aug_factor'] = 10
        tester_params['file_count'] = 100


    def _print_config():
        logger = logging.getLogger('root')
        logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
        [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


    ##########################################################################################

    if __name__ == "__main__":
        image = torch.load("data/" + str(i) + '.pt')
        # print('image', image)
        aug_score, use_time = main()
        score.append(aug_score)
        time_str.append(use_time)


print('score',np.mean(score))
print('time',np.mean(time_str))