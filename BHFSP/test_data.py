import torch
import numpy as np
import math
from scipy import stats
from pathlib import Path

def Generate(Job,Machine):
    Si = []
    for i in range(Job):
        S0 = []
        for j in range(len(Machine[0])): # stage
            machine_problem = rnd.randint(1, 10)
            S0 += [machine_problem for k in range(Machine[0, j])]
        Si.append(S0)
    operation_t = torch.tensor(Si)
    new_operation_t = torch.split(operation_t, Machine[0].tolist(), dim=1)
    list_operation_t = [new_operation_t[i].unsqueeze(0) for i in range(len(new_operation_t))]


    return list_operation_t


Job_list = [50]
machine_list = [[5,5,5,5,5]]

for i in range(50):
    rnd = np.random
    rnd.seed(i)
    for Job in Job_list:
        for machine_cnt_list in machine_list:
            stage_cnt = len(machine_cnt_list)
            Machine = np.array([[int(i) for i in machine_cnt_list]])
            PT = Generate(Job, Machine)  # time of jobs

            x = {'problems_INT_list': PT, 'problems_list': PT, 'batch_size': 1, 'stage_cnt': stage_cnt,
                 'machine_cnt_list': machine_cnt_list, 'job_cnt': Job, 'process_time_params':
                     {'time_low': 1, 'time_high': 10, 'distribution': 'uniform'}}

            path = Path('data/' + str(Job) + '/' + str(machine_cnt_list) + '/seed' + str(i) + '.pt')
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(x, path)





