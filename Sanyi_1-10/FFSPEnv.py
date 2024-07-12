
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

from dataclasses import dataclass
import torch
import itertools  # for permutation list

from FFSProblemDef import get_random_problems

# For Gantt Chart
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.transforms as mtransforms


@dataclass
class Reset_State:
    problems_list: list
    # len(problems_list) = stage_cnt
    # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
    # float type


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    #--------------------------------------
    step_cnt: int = 0
    stage_idx: torch.Tensor = None
    # shape: (batch, pomo)
    stage_machine_idx: torch.Tensor = None
    # shape: (batch, pomo)
    job_ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, job+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class FFSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.stage_cnt = env_params['stage_cnt']
        self.machine_cnt_list = env_params['machine_cnt_list']
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = env_params['job_cnt']
        self.process_time_params = env_params['process_time_params']
        self.pomo_size = env_params['pomo_size']
        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems_list = None
        # len(problems_list) = stage_cnt
        # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
        self.job_durations = None
        # shape: (batch, job+1, total_machine)
        # last job means NO_JOB ==> duration = 0

        # Dynamic
        ####################################
        self.time_idx = None
        # shape: (batch, pomo)
        self.sub_time_idx = None  # 0 ~ total_machine_cnt-1
        # shape: (batch, pomo)

        self.count_idx = None  # 0 ~ total_machine_cnt-1
        # shape: (batch, pomo)

        self.machine_idx = None  # must update according to sub_time_idx
        # shape: (batch, pomo)

        self.schedule = None
        # shape: (batch, pomo, machine, job+1)
        # records start time of each job at each machine
        self.machine_wait_step = None
        # shape: (batch, pomo, machine)
        # How many time steps each machine needs to run, before it become available for a new job

        self.machine_release = None
        # shape: (batch, pomo, machine)
        # Release time of blocked machine

        self.machine_count = None
        # shape: (batch, pomo, machine)
        # The number of completed job on the machine, if job_cnt, it means the machine is finished

        self.machine_location = None
        # shape: (batch, pomo, job+1)
        # index of machine each job can be processed at.

        self.job_location = None
        # shape: (batch, pomo, job+1)
        # index of stage each job can be processed at. if stage_cnt, it means the job is finished (when job_wait_step=0)
        self.job_wait_step = None
        # shape: (batch, pomo, job+1)
        # how many time steps job needs to wait, before it is completed and ready to start at job_location
        self.finished = None  # is scheduling done?
        # shape: (batch, pomo)

        # STEP-State
        ####################################
        self.step_state = None

        self.current_node = None

        self.job_machine_count = None
        # The machine that processes each job at each stage


    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_INT_list = get_random_problems(batch_size, self.stage_cnt, self.machine_cnt_list,
                                                self.job_cnt, self.process_time_params)

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

    def load_problems_manual(self, problems_INT_list):
        # problems_INT_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])

        self.batch_size = problems_INT_list[0].size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

    def reset(self):
        self.current_node = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.long, fill_value=-999999)

        self.time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.sub_time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.count_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)

        self.machine_idx = self.sm_indexer.get_machine_index(self.POMO_IDX, self.sub_time_idx)
        # shape: (batch, pomo)

        self.machine_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)

        self.schedule = torch.full(size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt+1),
                                   dtype=torch.long, fill_value=-999999)
        # shape: (batch, pomo, machine, job+1)
        self.machine_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)


        self.machine_release = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)

        self.machine_count = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)

        self.job_machine_count = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1, self.stage_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, job+1, stage)

        self.job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.finished = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)
        # shape: (batch, pomo)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)


        reward = None
        done = None
        return Reset_State(self.problems_list), reward, done

    def pre_step(self):
        self._update_step_state()
        self.step_state.step_cnt = 0
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, job_idx):
        # job_idx.shape: (batch, pomo)
        self.current_node = job_idx

        self.schedule[self.BATCH_IDX, self.POMO_IDX, self.machine_idx, job_idx] = self.time_idx

        job_length = self.job_durations[self.BATCH_IDX, job_idx, self.machine_idx]
        # shape: (batch, pomo)
        self.machine_wait_step[self.BATCH_IDX, self.POMO_IDX, self.machine_idx] = job_length
        # shape: (batch, pomo, machine)
        self.job_location[self.BATCH_IDX, self.POMO_IDX, job_idx] += 1
        # shape: (batch, pomo, job+1)
        self.job_wait_step[self.BATCH_IDX, self.POMO_IDX, job_idx] = job_length
        # shape: (batch, pomo, job+1)
        self.finished = (self.job_location[:, :, :self.job_cnt] == self.stage_cnt).all(dim=2)
        # shape: (batch, pomo)



        new_stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        # shape: (batch, pomo)

        # machine_job_bool = self.job_location == new_stage_idx[:,:,None]
        # shape: (batch, pomo, job+1)


        job_bool = (job_idx != self.job_cnt)

        job_release = self.machine_release[~job_bool]

        self.machine_release[self.BATCH_IDX, self.POMO_IDX, self.machine_idx] = 99999

        self.machine_release[~job_bool] = job_release




        machine_loaction = self.machine_location[self.BATCH_IDX, self.POMO_IDX, job_idx]
        machine_bool = (self.machine_idx >= self.machine_cnt_list[0]) & (job_idx != self.job_cnt)
        machine_release = self.machine_release[~machine_bool]
        self.machine_release[self.BATCH_IDX, self.POMO_IDX, machine_loaction] = self.time_idx
        self.machine_release[~machine_bool] = machine_release
        self.machine_release[:, :, -self.machine_cnt_list[0]:] = 0



        self.machine_location[self.BATCH_IDX, self.POMO_IDX, job_idx] = self.machine_idx
        # shape: (batch, pomo, job+1)


        # record the machine for each job
        self.job_machine_count[self.BATCH_IDX, self.POMO_IDX, job_idx, new_stage_idx] = self.machine_idx



        ####################################
        done = self.finished.all()

        if done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._move_to_next_machine()
            self._update_step_state()


        if done:
            reward = -self._get_makespan()  # Note the MINUS Sign ==> We want to MAXIMIZE reward
            # shape: (batch, pomo)
            # self.draw_Gantt_Chart(0,0,float(-reward[0,0].item()))
        else:
            reward = None

        return self.step_state, reward, done

    def _move_to_next_machine(self):

        b_idx = torch.flatten(self.BATCH_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        p_idx = torch.flatten(self.POMO_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        ready = torch.flatten(self.finished)
        # shape: (batch*pomo,) == (not_ready_cnt,)


        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        p_idx = p_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)

        while ~ready.all():
            new_sub_time_idx = self.sub_time_idx[b_idx, p_idx] + 1
            # shape: (not_ready_cnt,)
            step_time_required = new_sub_time_idx == self.total_machine_cnt
            # shape: (not_ready_cnt,)

            new_count_idx = self.count_idx[b_idx, p_idx] + 1
            count_required = new_count_idx == self.total_machine_cnt * 3

            self.time_idx[b_idx, p_idx] += count_required.long()
            new_sub_time_idx[step_time_required] = 0
            self.sub_time_idx[b_idx, p_idx] = new_sub_time_idx
            new_machine_idx = self.sm_indexer.get_machine_index(p_idx, new_sub_time_idx)
            self.machine_idx[b_idx, p_idx] = new_machine_idx

            new_count_idx[count_required] = 0
            self.count_idx[b_idx, p_idx] = new_count_idx


            machine_wait_steps = self.machine_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[count_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.machine_wait_step[b_idx, p_idx, :] = machine_wait_steps

            job_wait_steps = self.job_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[count_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.job_wait_step[b_idx, p_idx, :] = job_wait_steps

            machine_ready_1 = self.machine_wait_step[b_idx, p_idx, new_machine_idx] == 0
            # shape: (not_ready_cnt,)


            machine_ready_2 = self.machine_release[b_idx, p_idx, new_machine_idx] <= self.time_idx[b_idx, p_idx]

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)
            # shape: (not_ready_cnt,)

            job_ready_1 = (self.job_location[b_idx, p_idx, :self.job_cnt] == new_stage_idx[:, None])
            # shape: (not_ready_cnt, job)
            job_ready_2 = (self.job_wait_step[b_idx, p_idx, :self.job_cnt] == 0)
            # shape: (not_ready_cnt, job)


            job_ready = (job_ready_1 & job_ready_2).any(dim=1)
            # shape: (not_ready_cnt,)

            machine_ready = machine_ready_1 & machine_ready_2
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            p_idx = p_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)




    def _update_step_state(self):

        self.step_state.step_cnt += 1

        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        # shape: (batch, pomo)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.POMO_IDX, self.sub_time_idx)
        # shape: (batch, pomo)

        job_loc = self.job_location[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)
        job_wait_t = self.job_wait_step[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)

        job_in_stage = job_loc == self.step_state.stage_idx[:, :, None]
        # shape: (batch, pomo, job)
        job_not_waiting = (job_wait_t == 0)
        # shape: (batch, pomo, job)
        job_available = job_in_stage & job_not_waiting
        # shape: (batch, pomo, job)

        job_in_previous_stages = (job_loc < self.step_state.stage_idx[:, :, None]).any(dim=2)
        # shape: (batch, pomo)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(dim=2)
        # shape: (batch, pomo)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished
        # shape: (batch, pomo)

        self.step_state.job_ninf_mask = torch.full(size=(self.batch_size, self.pomo_size, self.job_cnt + 1),
                                                   fill_value=float('-inf'))

        # shape: (batch, pomo, job+1)
        job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
        # shape: (batch, pomo, job+1)


        self.step_state.job_ninf_mask[job_enable] = 0
        # shape: (batch, pomo, job+1)

        self.step_state.job_ninf_mask[:, :, -1] = float('-inf')
        # shape: (batch, pomo, job+1)


        newly_finished = (self.step_state.job_ninf_mask == float('-inf')).all(dim=2)
        self.step_state.job_ninf_mask[:, :, -1][newly_finished] = 0
        # shape: (batch, pomo, job+1)

        self.step_state.finished = self.finished
        # shape: (batch, pomo)

        # print('self.step_state.job_ninf_mask',self.step_state.job_ninf_mask)


    def _get_makespan(self):

        job_durations_perm = self.job_durations.permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, None, :, :]
        # shape: (batch, pomo, machine, job+1)

        end_time_max, _ = end_schedule[:, :, :, :self.job_cnt].max(dim=3)
        # shape: (batch, pomo, machine)
        end_time_max, _ = end_time_max.max(dim=2)
        # shape: (batch, pomo)

        return end_time_max

    def draw_Gantt_Chart(self, batch_i, pomo_i, score):

        config = {
            "font.family": 'serif',
            "font.size": 16,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False,
            'font.weight': 'bold',
        }


        sub_font = {'family': 'serif',
                    'size': 20,
                    }

        plt.rcParams.update(config)

        job_durations = self.job_durations[batch_i, :, :]
        # shape: (job, machine)
        schedule = self.schedule[batch_i, pomo_i, :, :]
        # shape: (machine, job)

        job_durations_perm = self.job_durations.permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        batch_end_schedule = self.schedule + job_durations_perm[:, None, :, :]
        # shape: (batch, pomo, machine, job+1)

        end_schedule = batch_end_schedule[batch_i, pomo_i, :, :]

        total_machine_cnt = self.total_machine_cnt
        makespan = self._get_makespan()[batch_i, pomo_i].item()

        # Create figure and axes
        fig,ax = plt.subplots(figsize=(20, 10))
        cmap = self._get_cmap(self.job_cnt)

        plt.xlim(0, makespan)
        plt.ylim(0, total_machine_cnt)
        ax.invert_yaxis()

        plt.plot([0, makespan], [self.machine_cnt_list[0], self.machine_cnt_list[0]], 'black')
        plt.plot([0, makespan], [self.machine_cnt_list[0] + self.machine_cnt_list[1], self.machine_cnt_list[0]+self.machine_cnt_list[1]], 'black')



        for machine_idx in range(total_machine_cnt):

            duration = job_durations[:, machine_idx]
            # shape: (job)
            machine_schedule = schedule[machine_idx, :]
            # shape: (job)
            end_machine_schedule = end_schedule[machine_idx, :]
            # shape: (job)

            for job_idx in range(self.job_cnt):

                job_length = duration[job_idx].item()
                job_start_time = machine_schedule[job_idx].item()
                job_end_time = end_machine_schedule[job_idx].item()

                all_machine = schedule[:, job_idx]
                positive_all_machine = all_machine[all_machine >= 0]

                if job_start_time >= 0:

                    start_job_index = torch.where(positive_all_machine == job_start_time)

                    if (start_job_index[0].size(0) == 1):
                        if (start_job_index[0][0] < self.stage_cnt - 1):
                            next_job_time = positive_all_machine[start_job_index[0][0] + 1]
                            rect_release = patches.Rectangle((job_end_time, machine_idx),
                                                             int(next_job_time - job_end_time), 1, edgecolor='black',
                                                             facecolor=cmap(job_idx), hatch='/')
                            ax.add_patch(rect_release)

                    elif (start_job_index[0].size(0) == 2):
                        if (start_job_index[0][1] < self.stage_cnt - 1) &  (machine_idx != 2)  &  (machine_idx != 10)  &  (machine_idx != 11) \
                                & (machine_idx != 12) &  (machine_idx != 13)  &  (machine_idx != 14) &  (machine_idx != 15)  &  (machine_idx != 41)\
                                & (machine_idx != 42) &  (machine_idx != 43)  &  (machine_idx != 45)  &  (machine_idx != 46)  &  (machine_idx != 47) \
                                &  (machine_idx != 48)  &  (machine_idx != 49)  &  (machine_idx != 51)  &  (machine_idx != 52)  &  (machine_idx != 53)\
                                &  (machine_idx != 54)  &  (machine_idx != 55):
                            next_job_time = positive_all_machine[start_job_index[0][1] + 1]
                            rect_release = patches.Rectangle((job_end_time, machine_idx),
                                                             int(next_job_time - job_end_time), 1, edgecolor='black',
                                                             facecolor=cmap(job_idx), hatch='/')
                            ax.add_patch(rect_release)
                        if (start_job_index[0][0] < self.stage_cnt - 1) & ((machine_idx == 2)  |  (machine_idx == 10)  |  (machine_idx == 11) \
                                | (machine_idx == 12) | (machine_idx == 13)  |  (machine_idx == 14) |  (machine_idx == 15)  | (machine_idx == 41)\
                                | (machine_idx == 42) |  (machine_idx == 43)  |  (machine_idx == 45)  |  (machine_idx == 46)  |  (machine_idx == 47) \
                                |  (machine_idx == 48)  |  (machine_idx == 49)  |  (machine_idx == 51)  | (machine_idx == 52)  |  (machine_idx == 53)\
                                |  (machine_idx == 54)  |  (machine_idx == 55)):
                            next_job_time = positive_all_machine[start_job_index[0][0] + 1]
                            rect_release = patches.Rectangle((job_end_time, machine_idx),
                                                             int(next_job_time - job_end_time), 1, edgecolor='black',
                                                             facecolor=cmap(job_idx), hatch='/')
                            ax.add_patch(rect_release)

                    # Create a Rectangle patch
                    rect = patches.Rectangle((job_start_time,machine_idx),job_length,1,edgecolor='black', facecolor=cmap(job_idx))
                    ax.add_patch(rect)


        process = ['Online track frame', 'Flipping track frame + Pre-install', 'Buffer','  Assembling rotary support', '  Tighten the rotary support ', '  Assembling walking motor', '  Assembling tensioning device', '   Assembling middle return joint', '  Connect the middle return hose ', ' Installing tracks +Online vehicle assembly', 'Buffer','  Coating with lubricating grease', '  Combination and lifting', '  Combination and fastening', '  Connecting lines', '  Connecting pipe clamp', '  Assembling the boom cylinder', '  Counterweight and Assembly', '  Assembling the cab', '  Assembling hand and foot valves', '  Connecting electrical appliances', '  Installing Interior', '  Installing seats', '  Static Filtering', 'Buffer', ' One time filtering', 'Buffer', '  Assembling work equipment', 'Buffer', '  Assembling the bucket']
        y = []
        for i in range(len(self.machine_cnt_list)):

                for j in range(self.machine_cnt_list[i]):
                    y.append(process[i])


        x = []
        for s in range(self.total_machine_cnt):
            x.append(s)
        plt.yticks(x, y)

        labels = ax.get_yticklabels()

        shift = 0.05

        for label in labels:
            offset = mtransforms.ScaledTranslation(0, -shift, plt.gcf().dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

        rect_release = patches.Rectangle((0, 0), 0, 1, edgecolor='black', facecolor=cmap(0), hatch='/',
                                         label='Blocking')
        ax.add_patch(rect_release)
        rect = patches.Rectangle((0, 0), 0, 1, edgecolor='black', facecolor=cmap(0), label='Normal')
        ax.add_patch(rect)

        plt.title("Sanyi-instance10(job=" + str(self.job_cnt) + ",stage=" + str(self.stage_cnt) + "), Object Value=" + str(score),fontdict=sub_font,fontweight='bold')

        ax.grid()
        ax.set_axisbelow(True)

        plt.legend()

        plt.savefig(
            '../result.jpg',
            dpi=500, bbox_inches='tight', pad_inches=0)

        plt.show()

    def _get_cmap(self, color_cnt):

        colors_list = ['red', 'orange', 'yellow', 'green', 'blue',
                       'purple', 'aqua', 'aquamarine', 'black',
                       'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate',
                       'coral', 'cornflowerblue', 'darkblue', 'darkgoldenrod', 'darkgreen',
                       'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                       'darkorchid', 'darkred', 'darkslateblue', 'darkslategrey', 'darkturquoise',
                       'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
                       'forestgreen', 'gold', 'goldenrod', 'gray', 'greenyellow',
                       'hotpink', 'indianred', 'khaki', 'lawngreen', 'magenta',
                       'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                       'mediumpurple',
                       'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                       'navy', 'olive', 'olivedrab', 'orangered',
                       'orchid',
                       'palegreen', 'paleturquoise', 'palevioletred', 'pink', 'plum', 'powderblue',
                       'rebeccapurple',
                       'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'sienna',
                       'silver', 'skyblue', 'slateblue',
                       'springgreen',
                       'steelblue', 'tan', 'teal', 'thistle',
                       'tomato', 'turquoise', 'violet', 'yellowgreen']

        cmap = ListedColormap(colors_list, N=color_cnt)

        return cmap


# # When test
class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        # assert env.machine_cnt_list == [4, 4, 4]
        # assert env.pomo_size == 24

        machine_SUBindex_0 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size, env.machine_cnt_list[0])
        machine_SUBindex_1 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[1])
        machine_SUBindex_2 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[2])
        machine_SUBindex_3 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[3])
        machine_SUBindex_4 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[4])
        machine_SUBindex_5 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[5])
        machine_SUBindex_6 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[6])
        machine_SUBindex_7 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[7])
        machine_SUBindex_8 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[8])
        machine_SUBindex_9 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[9])
        machine_SUBindex_10 = torch.tensor(list(itertools.permutations([0, 1, 2, 3, 4, 5])))
        machine_SUBindex_11 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[11])
        machine_SUBindex_12 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[12])
        machine_SUBindex_13 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                    env.machine_cnt_list[13])
        machine_SUBindex_14 = torch.tensor(list(itertools.permutations([0, 1, 2, 3]))).repeat(30,1)
        machine_SUBindex_15 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[15])
        machine_SUBindex_16 = torch.tensor(list(itertools.permutations([0, 1]))).repeat(360,1)
        machine_SUBindex_17 = torch.tensor(list(itertools.permutations([0, 1]))).repeat(360,1)
        machine_SUBindex_18 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[18])
        machine_SUBindex_19 = torch.tensor(list(itertools.permutations([0, 1, 2, 3, 4]))).repeat(6,1)
        machine_SUBindex_20 = torch.tensor(list(itertools.permutations([0, 1, 2]))).repeat(120,1)
        machine_SUBindex_21 = torch.tensor(list(itertools.permutations([0, 1]))).repeat(360,1)
        machine_SUBindex_22 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[22])
        machine_SUBindex_23 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[23])
        machine_SUBindex_24 = torch.tensor(list(itertools.permutations([0, 1, 2]))).repeat(120,1)
        machine_SUBindex_25 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[25])
        machine_SUBindex_26 = torch.tensor(list(itertools.permutations([0, 1, 2, 3, 4]))).repeat(6,1)
        machine_SUBindex_27 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[27])
        machine_SUBindex_28 = torch.tensor(list(itertools.permutations([0, 1, 2, 3, 4]))).repeat(6,1)
        machine_SUBindex_29 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
                                                                                     env.machine_cnt_list[29])


        self.machine_SUBindex_table = torch.cat((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2, machine_SUBindex_3,
                                                 machine_SUBindex_4, machine_SUBindex_5, machine_SUBindex_6, machine_SUBindex_7,
                                                 machine_SUBindex_8, machine_SUBindex_9, machine_SUBindex_10, machine_SUBindex_11,
                                                 machine_SUBindex_12, machine_SUBindex_13, machine_SUBindex_14, machine_SUBindex_15,
                                                 machine_SUBindex_16, machine_SUBindex_17, machine_SUBindex_18, machine_SUBindex_19,
                                                 machine_SUBindex_20, machine_SUBindex_21, machine_SUBindex_22, machine_SUBindex_23,
                                                 machine_SUBindex_24, machine_SUBindex_25, machine_SUBindex_26, machine_SUBindex_27,
                                                 machine_SUBindex_28, machine_SUBindex_29), dim=1)
        # machine_SUBindex_table.shape: (pomo, total_machine)

        starting_SUBindex = [0, 1,2,3,4,5,6,7, 8,9,10,16,17,18,19,23,24,26,28,29,34,37,39,40,41,44,45,50,51,56]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        machine_order_3 = machine_SUBindex_3 + starting_SUBindex[3]
        machine_order_4 = machine_SUBindex_4 + starting_SUBindex[4]
        machine_order_5 = machine_SUBindex_5 + starting_SUBindex[5]
        machine_order_6 = machine_SUBindex_6 + starting_SUBindex[6]
        machine_order_7 = machine_SUBindex_7 + starting_SUBindex[7]
        machine_order_8 = machine_SUBindex_8 + starting_SUBindex[8]
        machine_order_9 = machine_SUBindex_9 + starting_SUBindex[9]
        machine_order_10 = machine_SUBindex_10 + starting_SUBindex[10]
        machine_order_11 = machine_SUBindex_11 + starting_SUBindex[11]
        machine_order_12 = machine_SUBindex_12 + starting_SUBindex[12]
        machine_order_13 = machine_SUBindex_13 + starting_SUBindex[13]
        machine_order_14 = machine_SUBindex_14 + starting_SUBindex[14]
        machine_order_15 = machine_SUBindex_15 + starting_SUBindex[15]
        machine_order_16 = machine_SUBindex_16 + starting_SUBindex[16]
        machine_order_17 = machine_SUBindex_17 + starting_SUBindex[17]
        machine_order_18 = machine_SUBindex_18 + starting_SUBindex[18]
        machine_order_19 = machine_SUBindex_19 + starting_SUBindex[19]
        machine_order_20 = machine_SUBindex_20 + starting_SUBindex[20]
        machine_order_21 = machine_SUBindex_21 + starting_SUBindex[21]
        machine_order_22 = machine_SUBindex_22 + starting_SUBindex[22]
        machine_order_23 = machine_SUBindex_23 + starting_SUBindex[23]
        machine_order_24 = machine_SUBindex_24 + starting_SUBindex[24]
        machine_order_25 = machine_SUBindex_25 + starting_SUBindex[25]
        machine_order_26 = machine_SUBindex_26 + starting_SUBindex[26]
        machine_order_27 = machine_SUBindex_27 + starting_SUBindex[27]
        machine_order_28 = machine_SUBindex_28 + starting_SUBindex[28]
        machine_order_29 = machine_SUBindex_29 + starting_SUBindex[29]


        self.machine_table = torch.cat((machine_order_0, machine_order_1, machine_order_2, machine_order_3,
                                        machine_order_4, machine_order_5, machine_order_6, machine_order_7,
                                        machine_order_8, machine_order_9, machine_order_10, machine_order_11,
                                        machine_order_12, machine_order_13, machine_order_14, machine_order_15,
                                        machine_order_16, machine_order_17, machine_order_18, machine_order_19,
                                        machine_order_20, machine_order_21, machine_order_22, machine_order_23,
                                        machine_order_24, machine_order_25, machine_order_26, machine_order_27,
                                        machine_order_28, machine_order_29), dim=1)
        # machine_table.shape: (pomo, total_machine)

        # assert env.pomo_size == 1
        # self.machine_SUBindex_table = torch.tensor([[0,1,2,3,0,1,2,3,0,1,2,3]])
        # self.machine_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11]])

        self.stage_table = torch.tensor([0, 1, 2, 3, 4, 5,6,7, 8,9,10,10,10,10,10,10, 11, 12, 13, 14, 14, 14, 14,
                                         15, 16, 16, 17, 17, 18, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 22, 23,
                                         24, 24, 24, 25, 26, 26, 26, 26, 26, 27, 28, 28, 28, 28, 28, 29], dtype=torch.long)




# # When train
# class _Stage_N_Machine_Index_Converter:
#     def __init__(self, env):
#         # assert env.machine_cnt_list == [4, 4, 4]
#         # assert env.pomo_size == 24
#
#         machine_SUBindex_0 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size, env.machine_cnt_list[0])
#         machine_SUBindex_1 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[1])
#         machine_SUBindex_2 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[2])
#         machine_SUBindex_3 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[3])
#         machine_SUBindex_4 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[4])
#         machine_SUBindex_5 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[5])
#         machine_SUBindex_6 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[6])
#         machine_SUBindex_7 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[7])
#         machine_SUBindex_8 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[8])
#         machine_SUBindex_9 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[9])
#         machine_SUBindex_10 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_11 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[11])
#         machine_SUBindex_12 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[12])
#         machine_SUBindex_13 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                     env.machine_cnt_list[13])
#         machine_SUBindex_14 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_15 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[15])
#         machine_SUBindex_16 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_17 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_18 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[18])
#         machine_SUBindex_19 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_20 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_21 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_22 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[22])
#         machine_SUBindex_23 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[23])
#         machine_SUBindex_24 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_25 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[25])
#         machine_SUBindex_26 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_27 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[27])
#         machine_SUBindex_28 = torch.tensor(list(itertools.permutations([0, 1, 2])))
#         machine_SUBindex_29 = torch.tensor(list(itertools.permutations([0]))).expand(env.pomo_size,
#                                                                                      env.machine_cnt_list[29])
#
#
#         self.machine_SUBindex_table = torch.cat((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2, machine_SUBindex_3,
#                                                  machine_SUBindex_4, machine_SUBindex_5, machine_SUBindex_6, machine_SUBindex_7,
#                                                  machine_SUBindex_8, machine_SUBindex_9, machine_SUBindex_10, machine_SUBindex_11,
#                                                  machine_SUBindex_12, machine_SUBindex_13, machine_SUBindex_14, machine_SUBindex_15,
#                                                  machine_SUBindex_16, machine_SUBindex_17, machine_SUBindex_18, machine_SUBindex_19,
#                                                  machine_SUBindex_20, machine_SUBindex_21, machine_SUBindex_22, machine_SUBindex_23,
#                                                  machine_SUBindex_24, machine_SUBindex_25, machine_SUBindex_26, machine_SUBindex_27,
#                                                  machine_SUBindex_28, machine_SUBindex_29), dim=1)
#         # machine_SUBindex_table.shape: (pomo, total_machine)
#
#         starting_SUBindex = [0, 1,2,3,4,5,6,7, 8,9,10,13,14,15,16,19,20,23,26,27,30,33,36,37,38,41,42,45,46,49]
#         machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
#         machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
#         machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
#         machine_order_3 = machine_SUBindex_3 + starting_SUBindex[3]
#         machine_order_4 = machine_SUBindex_4 + starting_SUBindex[4]
#         machine_order_5 = machine_SUBindex_5 + starting_SUBindex[5]
#         machine_order_6 = machine_SUBindex_6 + starting_SUBindex[6]
#         machine_order_7 = machine_SUBindex_7 + starting_SUBindex[7]
#         machine_order_8 = machine_SUBindex_8 + starting_SUBindex[8]
#         machine_order_9 = machine_SUBindex_9 + starting_SUBindex[9]
#         machine_order_10 = machine_SUBindex_10 + starting_SUBindex[10]
#         machine_order_11 = machine_SUBindex_11 + starting_SUBindex[11]
#         machine_order_12 = machine_SUBindex_12 + starting_SUBindex[12]
#         machine_order_13 = machine_SUBindex_13 + starting_SUBindex[13]
#         machine_order_14 = machine_SUBindex_14 + starting_SUBindex[14]
#         machine_order_15 = machine_SUBindex_15 + starting_SUBindex[15]
#         machine_order_16 = machine_SUBindex_16 + starting_SUBindex[16]
#         machine_order_17 = machine_SUBindex_17 + starting_SUBindex[17]
#         machine_order_18 = machine_SUBindex_18 + starting_SUBindex[18]
#         machine_order_19 = machine_SUBindex_19 + starting_SUBindex[19]
#         machine_order_20 = machine_SUBindex_20 + starting_SUBindex[20]
#         machine_order_21 = machine_SUBindex_21 + starting_SUBindex[21]
#         machine_order_22 = machine_SUBindex_22 + starting_SUBindex[22]
#         machine_order_23 = machine_SUBindex_23 + starting_SUBindex[23]
#         machine_order_24 = machine_SUBindex_24 + starting_SUBindex[24]
#         machine_order_25 = machine_SUBindex_25 + starting_SUBindex[25]
#         machine_order_26 = machine_SUBindex_26 + starting_SUBindex[26]
#         machine_order_27 = machine_SUBindex_27 + starting_SUBindex[27]
#         machine_order_28 = machine_SUBindex_28 + starting_SUBindex[28]
#         machine_order_29 = machine_SUBindex_29 + starting_SUBindex[29]
#
#
#         self.machine_table = torch.cat((machine_order_0, machine_order_1, machine_order_2, machine_order_3,
#                                         machine_order_4, machine_order_5, machine_order_6, machine_order_7,
#                                         machine_order_8, machine_order_9, machine_order_10, machine_order_11,
#                                         machine_order_12, machine_order_13, machine_order_14, machine_order_15,
#                                         machine_order_16, machine_order_17, machine_order_18, machine_order_19,
#                                         machine_order_20, machine_order_21, machine_order_22, machine_order_23,
#                                         machine_order_24, machine_order_25, machine_order_26, machine_order_27,
#                                         machine_order_28, machine_order_29), dim=1)
#         # machine_table.shape: (pomo, total_machine)
#
#
#         self.stage_table = torch.tensor([0, 1, 2, 3, 4, 5,6,7, 8,9,10,10,10, 11, 12, 13, 14, 14, 14,
#                                          15, 16, 16, 16, 17, 17, 17, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 23,
#                                          24, 24, 24, 25, 26, 26, 26, 27, 28, 28, 28, 29], dtype=torch.long)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, POMO_IDX, sub_time_idx):
        # POMO_IDX.shape: (batch, pomo)
        # sub_time_idx.shape: (batch, pomo)
        return self.machine_table[POMO_IDX, sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]
