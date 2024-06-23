
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

for i in range(50):
    env_params = {
        'stage_cnt': 7, # Change 1
        'machine_cnt_list': [3,1,3,1,3,1,3],# Change 2
        'job_cnt': 50, # Change 3
        'process_time_params': {
            'time_low': 1,
            'time_high': 10,
        },
        'pomo_size': 6,  # Change 4
        # assuming 4 machines at each stage! 4*3*2*1
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
        'one_hot_seed_cnt': 6,  # must be >= machine_cnt
    }

    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            # 'path': './result/20240421_105615_matnet_train10_machine[3, 1, 3]', # Change 5
            # 'path': './result/20240422_164735_matnet_train20_machine[3, 1, 3]',
            # 'path': './result/20240419_235157_matnet_train50_machine[3, 1, 3]',
            # 'path': './result/20240421_133019_matnet_train100_machine[3, 1, 3]',
            #
            # 'path': './result/20240422_111134_matnet_train10_machine[4, 1, 4]',
            # 'path': './result/20240421_134355_matnet_train20_machine[4, 1, 4]',
            # 'path': './result/20240421_133946_matnet_train50_machine[4, 1, 4]',
            # 'path': './result/20240423_111038_matnet_train100_machine[4, 1, 4]',
            #
            # 'path': './result/20240422_201958_matnet_train10_machine[5, 1, 5]',
            # 'path': './result/20240421_183207_matnet_train20_machine[5, 1, 5]',
            # 'path': './result/20240423_110930_matnet_train50_machine[5, 1, 5]',
            # 'path': './result/20240508_105308_matnet_train100_machine[5, 1, 5]',
            #
            # 'path': './result/20240617_235217_matnet_train50_machine[3, 1, 3, 1, 3]',
            'path': './result/20240422_212751_matnet_train50_machine[3, 1, 3, 1, 3, 1, 3]',
            # 'path': './result/20240422_221140_matnet_train50_machine[3, 1, 3, 1, 3, 1, 3, 1, 3]',
            #
            # 'path': './result/20240618_123739_matnet_train50_machine[4, 1, 4, 1, 4]',
            # 'path': './result/20240422_215907_matnet_train50_machine[4, 1, 4, 1, 4, 1, 4]',
            # 'path': './result/20240422_221125_matnet_train50_machine[4, 1, 4, 1, 4, 1, 4, 1, 4]',

            # 'path': './result/20240425_084345_matnet_train50_machine[5, 1, 5, 1, 5]',
            # 'path': './result/20240430_173755_matnet_train50_machine[5, 1, 5, 1, 5, 1, 5]',
            # 'path': './result/20240501_113031_matnet_train50_machine[5, 1, 5, 1, 5, 1, 5, 1, 5]',

            # directory path of pre-trained model and log files saved.
            'epoch': 100,  # epoch version of pre-trained model to load.
        },
        'saved_problem_folder': "data/" +str(env_params['job_cnt'])+'/'+ str(env_params['machine_cnt_list']) + '/',
        'saved_problem_filename': 'seed' + str(i) + '.pt',
        'problem_count': 1,
        'test_batch_size':1,
        'augmentation_enable': True,
        'aug_factor': 128,
        'aug_batch_size': 300,
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    logger_params = {
        'log_file': {
            'desc': 'ffsp_matnet_test_job'+str(env_params['job_cnt'])+'_machine'+
                    str(env_params['machine_cnt_list'])+'_aug_'+str(tester_params['aug_factor']),
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
        image = torch.load("data/" +str(env_params['job_cnt'])+'/'+ str(env_params['machine_cnt_list']) + '/'+'seed' + str(i) + '.pt')
        aug_score, use_time = main()
        score.append(aug_score)
        time_str.append(use_time)


print('score',np.mean(score)) # Average objective value
print('time',np.mean(time_str)) # Average time