import torch
import numpy as np
import math
from scipy import stats
from pathlib import Path

def Generate(Job, s, Machine):
    Si = []  # Job i
    for i in range(Job[s]):
        S0 = []
        Schedule = [
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 19, 6, 4, 4, 5, 17, 24, 6, 4, 3, 0, 5, 0, 6, 0, 5, 0, 5],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 3, 5, 4, 0, 2, 2, 2, 20, 6, 4, 3, 5, 17, 24, 5, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 3, 5, 4, 0, 2, 2, 2, 20, 6, 4, 4, 5, 17, 23, 6, 4, 3, 0, 5, 0, 5, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 3, 5, 4, 0, 2, 2, 2, 20, 6, 4, 3, 5, 17, 24, 5, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 19, 6, 4, 4, 5, 17, 23, 5, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 19, 6, 4, 4, 5, 17, 23, 6, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 1, 2, 2, 20, 6, 4, 4, 5, 17, 23, 6, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 20, 6, 4, 4, 5, 17, 23, 6, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 3, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 20, 6, 4, 4, 5, 17, 24, 5, 4, 3, 0, 5, 0, 6, 0, 5, 0, 4],
            [3, 1, 4, 1, 0, 4, 1, 3, 1, 2, 5, 4, 0, 2, 2, 2, 19, 6, 4, 4, 5, 17, 23, 5, 4, 3, 0, 5, 0, 6, 0, 5, 0,
             4]]  # 11-20

        for j in range(len(Machine[0])): # stage
            S0 += [Schedule[s][j] for k in range(Machine[0, j])]
        Si.append(S0)
    operation_t = torch.tensor(Si)

    new_operation_t = torch.split(operation_t, Machine[0].tolist(), dim=1)
    list_operation_t = [new_operation_t[i].unsqueeze(0) for i in range(len(new_operation_t))]


    return list_operation_t


Job_list = [42, 60, 50, 120, 104, 114, 156, 168, 200, 188]
machine_list = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 3, 1, 1, 1, 1, 4, 3, 1, 1, 1, 3, 1, 5, 1, 5, 1, 5, 1]]

for i in range(len(Job_list)):
    for machine_cnt_list in machine_list:
        stage_cnt = len(machine_cnt_list)
        Machine = np.array([[int(i) for i in machine_cnt_list]])
        PT = Generate(Job_list, i, Machine)  # Time of jobs

        x = {'problems_INT_list': PT, 'problems_list': PT, 'batch_size': 1, 'stage_cnt': stage_cnt,
             'machine_cnt_list': machine_cnt_list, 'job_cnt': Job_list[i], 'process_time_params':
                 {'time_low': 1, 'time_high': 10, 'distribution': 'uniform'}}

        path = Path('data/' + str(i) + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(x, path)








