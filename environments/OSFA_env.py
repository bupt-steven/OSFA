#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
# from environments.basic_class.state_class import state
from environments.basic_class.pystream import *
import logging
from utility.util import *

INFO = 1
DEBUG = 2
PLOT = 3

verbose = DEBUG

# data mining,cache,web search,hadoop
sc = [1, 10, 2, 2]
cdfx = [[0, 0.5, 0.8, 1], [0, 0.35, 0.6, 0.804, 0.99, 1], [0, 0.203, 0.53, 1], [0, 0.104, 0.94, 1]]
cdfy = [[0, 0.05, 1.02, 6], [0, 0.18, 1.98, 3.3, 3.6, 4.71], [0, 1.296, 1.906, 5.7], [0, 1.554, 2.872, 5.7]]
multi_factor = 1e6

class OSFA:
    def __init__(
            self,
            n_ToR_per_unit,
            m_level,
            k_regular,
            distribution_id,
            # routing_strategy,
            file_path,
            total_size,
            change_interval=10,
            alpha=300,
            factor=2,
            global_view=True,
            coded_action=True,
            single_model=False,
    ):
        # ------------------------------------- basic parameters -------------------------------------------------------

        self.alpha = alpha  # reward factor
        self.factor = factor  # K_regular = n_ToR_per_unit / factor
        self.n_ToR_per_unit = n_ToR_per_unit  # n : num of ToR per unit
        self.m_level = m_level  # m : recursive level, single unit is defined as m = 0
        self.k_regular = k_regular  # k : default as n/2
        self.change_interval = change_interval  # T : interval of topology change
        self.distribution_id = distribution_id
        self.file_path = file_path
        self.total_size = total_size

        # ------------------------------------- control flags ----------------------------------------------------------
        # agent
        self.single_model = single_model  # True: single big OCS for each unit;     False: k small OCSs with funID

        # state
        self.global_view = global_view  # True: with source permutation;     False: only extract related source

        # action
        self.coded_action = coded_action  # True: exchange every 2 link;        False: directly output topology_ID

        # reward

        # ------------------------------------- quantity computation ---------------------------------------------------

        if self.single_model:
            # this OCS port is a sum of ports in a single unit
            # here assume that a big OCS with enough ports is existing.
            self.n_OCS_port = int(n_ToR_per_unit * k_regular)  # total port nunber of OCS
            self.num_OCS = int((m_level + 1) * pow(n_ToR_per_unit, m_level))  # total number of OCS
            self.num_per_level_OCS = int(pow(n_ToR_per_unit, m_level))  # number of OCS for each level
        else:
            self.n_OCS_port = n_ToR_per_unit  # port number of OCS
            self.num_OCS = int((m_level + 1) * pow(n_ToR_per_unit, m_level) * k_regular)  # total number of OCS
            self.num_per_level_OCS = int(pow(n_ToR_per_unit, m_level)) * k_regular  # number of OCS for each level

        self.num_ToR = int(pow(n_ToR_per_unit, m_level + 1))  # total number of ToR
        self.num_ToR_port = 2 * (m_level + 1) * k_regular  # port number of ToR (including port connected to servers)
        self.num_server = int((m_level + 1) * pow(n_ToR_per_unit, m_level + 1) * k_regular)  # total number of server

        self.num_per_OCS_link = int(self.n_OCS_port / 2)  # number of link for each OCS
        # ------------------------------------- action related ---------------------------------------------------------
        # todo : to be added
        # 这里有巨大的问题！
        if self.coded_action:
            # it means that input state must contain topology/link state
            self.num_actions = int(((self.n_OCS_port ** 2) / 8 - self.n_OCS_port / 4) * 2 + 1)  # C^{2}_{\frac{n}{2}}
            self.action_space = [i for i in range(self.num_actions)]  # * 2 means 2 modes to exchange links
        else:
            # todo 这个有问题！
            pass
            # self.num_actions = 1
            # for i in range(1, self.n_OCS_port, 2):
            #     self.num_actions *= i                                                           # (n-1)!

        # ------------------------------------- member variable --------------------------------------------------------

        self.topology = []  # topology              size:[self.num_ToR, self.num_ToR]

        self.current_demand = []  # current demand        size:[self.num_ToR, self.num_ToR]
        self.normalized_demand = []
        self.standard_demand_memory = []
        self.normalized_demand_memory = []
        # self.next_demand = []                 # next demand           size:[self.num_ToR, self.num_ToR]

        self.observation_shape = [self.num_ToR, self.num_ToR]
        # observation is the permuted demand in each OCS's view
        self.current_D_observations = []  # current observations  size:[self.num_OCS, self.num_ToR, self.num_ToR]
        # self.next_D_observations = []         # next observations     size:[self.num_OCS, self.num_ToR, self.num_ToR]

        self.current_T_observations = []  # current observations  size:[self.num_OCS, self.num_ToR, self.num_ToR]
        # self.next_T_observations = []         # next observations     size:[self.num_OCS, self.num_ToR, self.num_ToR]

        self.current_time = 0  # current time          size:[1]  dtype:int

        # ------------------------------------- table initializations --------------------------------------------------

        self.ToR_ID_table = []  # ToR ID
        self.ToR_ID_table_create()  # seems no use
        if verbose >= DEBUG:
            logging.debug("--------------------ToR_ID_table--------------------")
            logging.debug(self.ToR_ID_table)

        self.OCS_ID_table = []  # map each OCS to ToR_ID that connected to it + function_ID
        self.OCS_ID_table_create()  # size: [self.num_OCS, self.n_OCS_port+1]
        if verbose >= DEBUG:
            logging.debug("--------------------OCS_ID_table--------------------")
            logging.debug(self.OCS_ID_table)  # dtype = pd.DataFrame

        self.ToR4OCS_ID_table = []  # map each ToR to OCS_ID that connected to it
        self.ToR4OCS_ID_table_create()  # size: [self.num_ToR, int(self.num_ToR_port/2)]
        if verbose >= DEBUG:
            logging.debug("--------------------ToR4OCS_ID_table--------------------")
            logging.debug(self.ToR4OCS_ID_table)  # dtype = pd.DataFrame

        self.permutation_table = []  # map each OCS to permuted demand in their own view
        self.permutation_table_create()  # size: [self.num_OCS, self.num_ToR]
        if verbose >= DEBUG:
            logging.debug("--------------------permutation_table--------------------")
            logging.debug(self.permutation_table)  # dtype = pd.DataFrame

        if self.coded_action:
            self.coded_action_table = []  # map coded_action_id to a link pair
            self.coded_action_table_create()  # size: [self.num_actions,2]
            if verbose >= DEBUG:
                logging.debug("--------------------coded_action_table--------------------")
                logging.debug(self.coded_action_table)  # dtype = pd.DataFrame

        self.link_port_table = []  # map link_id to end-to-end port pair
        # initialed in topology_init()
        # self.link_port_table_create()         # size: [self.num_OCS, self.num_per_OCS_link, 2]
        # print(self.link_port_table)           #

        # self.OCS_fun_ID_table = []
        self.OCS_fun_ID_table = self.OCS_ID_table.iloc[:, self.n_OCS_port]
        # ------------------------------------- real initializations ---------------------------------------------------
        # self.demand_memory_setup(0, [], self.file_path)
        # self.reset(0)
        if verbose == DEBUG:
            logging.debug("--------------------init topology--------------------")
            logging.debug(self.topology)
            logging.debug("--------------------link_port_table--------------------")
            logging.debug(self.link_port_table)
            logging.debug("--------------------init demand--------------------")
            logging.debug(self.current_demand)
            logging.debug("--------------------init normalized demand--------------------")
            logging.debug(self.normalized_demand)

        if verbose >= PLOT:
            # fig = plt.figure()
            fig, axarr = plt.subplots(2)
            ax0 = fig.add_subplot(121)
            im0 = ax0.imshow(self.topology)
            plt.colorbar(im0, fraction=0.046, pad=0.04)

            # action = np.random.randint(0, self.num_actions, self.num_OCS)
            action = np.ones([self.num_OCS], dtype=int)
            # print("--------------------action--------------------")
            # print(action)
            topo = self.action2topo(action)
            action = np.ones([self.num_OCS], dtype=int)*2
            topo = self.action2topo(action)
            # logging.info("--------------------new topo--------------------")
            # logging.info(topo)
            # ax1 = fig.add_subplot(122)
            # im1 = ax1.imshow(topo)
            # plt.colorbar(im1, fraction=0.046, pad=0.04)
            plt.show()

        if verbose >= INFO:
            logging.info("--------------------OSFA environment create!--------------------")
            logging.info(
                "ToR_per_unit: %d  n_OCS_port: %d  m_level: %d  k_regular: %d" % (self.n_ToR_per_unit, self.n_OCS_port,
                                                                                  self.m_level, self.k_regular))
            logging.info("num_OCS: %d  num_ToR: %d  num_ToR_port: %d  num_server: %d" % (self.num_OCS, self.num_ToR,
                                                                                  self.num_ToR_port, self.num_server))

    def reset(self, demand_id):
        self.current_time = 0
        self.topology_init()

        self.demand_load(demand_id)

        self.current_D_observations = self.permutation(self.normalized_demand)
        self.current_T_observations = self.permutation(self.topology)
        streamMain(0, self.topology, self.current_demand, 0)

        return self.current_D_observations, self.current_T_observations

    # init demand memory
    def demand_memory_setup(self, sample_num, sample_index, file_path, read_from_file=False):
        # read from file
        if read_from_file:
            self.demand_read_n_memory_setup(sample_index=sample_index, path=file_path)
        # generate and save
        else:
            self.demand_save_n_memory_setup(sample_num=sample_num, path=file_path, distribution_id=self.distribution_id)

    def demand_save_n_memory_setup(self, sample_num, path, distribution_id):
        self.standard_demand_memory = np.zeros([sample_num, self.num_ToR, self.num_ToR])
        self.normalized_demand_memory = np.zeros([sample_num, self.num_ToR, self.num_ToR])
        for i in range(sample_num):
            standard_d, normalized_d = self.demand_generate(distribution_id)
            self.standard_demand_memory[i][:] = standard_d
            self.normalized_demand_memory[i][:] = normalized_d
            to_file_path_s = path + "s_demand" + str(i) + ".txt"
            to_file_path_n = path + "n_demand" + str(i) + ".txt"
            util.writefile(standard_d, to_file_path_s)
            util.writefile(normalized_d, to_file_path_n)
            if i % 50 == 0:
                logging.info("successfully write %dth demand to file" % i)
        logging.info("successfully write all demands to file")

    def demand_read_n_memory_setup(self, sample_index, path):
        sample_num = len(sample_index)
        self.standard_demand_memory = np.zeros([sample_num, self.num_ToR, self.num_ToR])
        self.normalized_demand_memory = np.zeros([sample_num, self.num_ToR, self.num_ToR])
        for i in sample_index:
            file_path_s = path + "s_demand" + str(i) + ".txt"
            file_path_n = path + "n_demand" + str(i) + ".txt"
            self.standard_demand_memory[i][:] = readfile(file_path_s)
            self.normalized_demand_memory[i][:] = readfile(file_path_n)
            if i % 20 == 0:
                logging.info("successfully read %dth demand to file" % i)
        logging.info("successfully read all demands to file")

    def step(self, action):
        last_topo = copy.deepcopy(self.topology)
        # link_port_table_copy = copy.deepcopy(self.link_port_table)
        current_topo = self.action2topo(action)
        # print(current_topo)
        demand = self.current_demand

        # 不判断连通性
        spendTime, next_demand, _ = streamMain1(self.change_interval, current_topo, demand, 1)
        unconnected = False
        # 判断连通性
        # spendTime, next_demand, _, unconnected = streamMain(self.change_interval, current_topo, demand, 1)
        demand_sum = sum(sum(next_demand))
        logging.debug("res_demand: "+str(demand_sum))
        # print(spendTime)

        self.current_time += spendTime
        #
        if demand_sum > 0:
            done = False
        else:
            done = True
        # next_demand, current_time, done = self.run_time(current_topo, demand, self.change_interval)

        # todo: how to set reward
        if done:
            # reward = self.alpha * 1 / self.current_time
            reward = -spendTime
            # reward = 0
        else:
            if unconnected:
                logging.debug("it is disconnected!")
                current_topo = last_topo
                self.topology = last_topo
                self.link_port_table = link_port_table_copy
                reward = -100000
            else:
                reward = -spendTime

        self.current_demand = next_demand
        self.normalized_demand = next_demand / multi_factor
        next_D_observations = self.permutation(self.normalized_demand)
        self.current_D_observations = next_D_observations
        next_T_observations = self.permutation(current_topo)
        self.current_T_observations = next_T_observations
        return next_D_observations, next_T_observations, reward, done , unconnected

    def run_time(self, current_topo, current_demand, change_interval):
        next_demand = current_demand
        # standard_topo = [[0,1,1,2],[1,0,2,1],[1,2,0,1],[2,1,1,0]]
        standard_topo = [[0, 1, 2, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0],
                         [1, 0, 1, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0],
                         [2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0],
                         [1, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 2, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 2, 0, 0],
                         [0, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
                         [0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2],
                         [2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0],
                         [0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0],
                         [0, 0, 2, 0, 0, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1],
                         [0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2],
                         [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 1, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 0]]
        # print(standard_topo)
        for i in range(4):
            for j in range(4):
                if standard_topo[i][j] == current_topo[i][j]:
                    done = True
                else:
                    done = False
                    break

        current_time_ = random.randint(1, 10)
        # if random.random() > 0.9:
        #     done = True
        # else:
        #     done = False
        return next_demand, current_time_, done

    # init topology
    # different init strategy may lead to different learning result
    # here, in each OCS, the ith ToR are connected to i+(1+2*k)th ToR, where K in range(k_regular)
    def topology_init(self):
        topo = np.zeros([self.num_ToR, self.num_ToR])
        topo_fix = np.zeros([self.num_ToR, self.num_ToR])
        link_port_table = np.zeros([self.num_OCS, self.num_per_OCS_link, 2], dtype=int)

        for index, row in self.OCS_ID_table.iterrows():

            if self.single_model:
                link_count = 0
                for ToR in range(self.n_ToR_per_unit):
                    # fixed link
                    topo_fix[row[ToR]][row[(ToR + 1) % self.n_OCS_port]] = 1
                    topo_fix[row[(ToR + 1) % self.n_OCS_port]][row[ToR]] = 1
                    temp = ToR + 1
                    while temp < self.n_ToR_per_unit:
                        topo[row[ToR]][row[temp]] += 1
                        topo[row[temp]][row[ToR]] += 1
                        link_port_table[index][link_count][0] = row[ToR]
                        link_port_table[index][link_count][1] = row[temp]
                        temp += self.factor
                        link_count += 1
            else:
                k = index % self.k_regular
                link_count = 0
                for ToR in range(0, self.n_ToR_per_unit):
                    # fixed link
                    topo_fix[row[ToR]][row[(ToR + 1) % self.n_OCS_port]] = 1
                    topo_fix[row[(ToR + 1) % self.n_OCS_port]][row[ToR]] = 1
                    if k == 0:
                        temp = ToR + 1
                        while temp < self.n_ToR_per_unit:
                            topo[row[ToR]][row[temp]] += 1
                            topo[row[temp]][row[ToR]] += 1
                            temp += self.factor
                    if ToR % 2 == 00:
                        link_port_table[index][link_count][0] = row[ToR]
                        link_port_table[index][link_count][1] = row[(ToR + 2 * k + 1) % self.n_ToR_per_unit]
                        link_count += 1

        self.topology = topo + topo_fix
        self.link_port_table = link_port_table


    # convert list of actions to a full topology
    # action : size:[self.num_OCS]
    def action2topo(self, action):
        topo = self.topology
        for i in range(self.num_OCS):
            if self.coded_action:
                OCS_id = i
                # print("OCS_id: "+ str(OCS_id))
                action_id = action[i]
                # print("action_id: "+ str(action_id))
                link_pair = self.coded_action_table.iloc[action_id][0:2]
                # print("link_pair: " )
                # print(link_pair)
                exchange_mode = self.coded_action_table.iloc[action_id][2]
                port_pair = self.link_port_table[OCS_id][link_pair[0:2]]
                # print("port_pair: ")
                # print(port_pair)
                # print(topo)
                l1 = port_pair[0][0]
                r1 = port_pair[0][1]
                l2 = port_pair[1][0]
                r2 = port_pair[1][1]

                if exchange_mode != 2:
                    topo[l1][r1] -= 1
                    topo[l2][r2] -= 1
                    topo[r1][l1] -= 1
                    topo[r2][l2] -= 1

                    if exchange_mode == 0:
                        topo[l1][r2] += 1
                        topo[l2][r1] += 1
                        topo[r2][l1] += 1
                        topo[r1][l2] += 1

                        self.link_port_table[OCS_id][link_pair[0]][0:2] = [l1, r2]
                        self.link_port_table[OCS_id][link_pair[1]][0:2] = [l2, r1]
                    elif exchange_mode == 1:

                        topo[l1][l2] += 1
                        topo[r1][r2] += 1
                        topo[l2][l1] += 1
                        topo[r2][r1] += 1

                        self.link_port_table[OCS_id][link_pair[0]][0:2] = [l1, l2]
                        self.link_port_table[OCS_id][link_pair[1]][0:2] = [r1, r2]
            else:
                logging.ERROR("Sorry, it is empty here.")
        self.topology = topo
        return topo

    # resort source(demand,topo) according to the location of OCS
    # two working mode: local view  / global view (default)
    def permutation(self, source):
        observations = np.zeros([self.num_OCS, self.num_ToR, self.num_ToR])
        for i in range(self.num_OCS):
            if self.global_view:
                temp_permutation_table = self.permutation_table.iloc[i]
                for j in range(self.num_ToR):
                    for k in range(self.num_ToR):
                        observations[i][j][k] = source[temp_permutation_table[j]][temp_permutation_table[k]]
            else:
                row = self.OCS_ID_table.iloc[i]
                for ToR in range(self.n_ToR_per_unit):
                    observations[i][row[ToR], :] = source[row[ToR], :]
                    observations[i][:, row[ToR]] = source[:, row[ToR]]

        return observations

    # create table to store IDs for every ToR
    def ToR_ID_table_create(self):
        ID_table_shape = [self.n_ToR_per_unit for i in range(self.m_level + 1)]
        self.ToR_ID_table = np.arange(self.num_ToR).reshape(ID_table_shape)

    # create table to store ToR ID for every OCS
    def OCS_ID_table_create(self):
        OCS_ID_table = np.zeros([self.num_OCS, self.n_OCS_port + 1])

        # per_level_OCS = pow(self.n_ToR_per_unit, self.m_level)
        for level in range(self.m_level + 1):
            for i in range(self.num_per_level_OCS * level, self.num_per_level_OCS * (level + 1)):
                for j in range(self.n_OCS_port):
                    if self.single_model:
                        OCS_ID_table[i][j] = ((i - level * self.num_per_level_OCS) % pow(self.n_ToR_per_unit, level)) + \
                                             math.floor((i - level * self.num_per_level_OCS) / pow(self.n_ToR_per_unit,
                                                                                                   level)) * \
                                             pow(self.n_ToR_per_unit, level + 1) + \
                                             (j % self.n_ToR_per_unit) * pow(self.n_ToR_per_unit, level)
                        # function_ID
                        OCS_ID_table[i][self.n_OCS_port] = 0
                    else:
                        ii = math.floor((i - level * self.num_per_level_OCS) / self.k_regular)
                        temp = i
                        i = math.floor(i / self.k_regular)
                        for k in range(self.k_regular):
                            OCS_ID_table[self.k_regular * i + k][j] = (ii % pow(self.n_ToR_per_unit, level)) + \
                                                                      math.floor(ii / pow(self.n_ToR_per_unit, level)) * \
                                                                      pow(self.n_ToR_per_unit, level + 1) + \
                                                                      j * pow(self.n_ToR_per_unit, level)
                            # function_ID
                            OCS_ID_table[self.k_regular * i + k][self.n_OCS_port] = k
                        i = temp

        columns = []
        for i in range(self.n_OCS_port):
            columns.append("ToR" + str(i))
        columns.append("funID")
        self.OCS_ID_table = pd.DataFrame(OCS_ID_table, columns=columns, dtype=int)
        # self.OCS_ID_table = OCS_ID_table

    # create table to store OCS ID for every ToR
    def ToR4OCS_ID_table_create(self):
        ToR4OCS_ID_table = np.zeros([self.num_ToR, int(self.num_ToR_port / 2)])
        count = np.zeros(self.num_ToR, dtype=int)
        for index, row in self.OCS_ID_table.iterrows():
            if self.single_model:
                for i in row[:-1]:
                    ToR4OCS_ID_table[i][count[i]] = index
                    count[i] += 1
            else:
                for i in row[:-1]:
                    ToR4OCS_ID_table[i][count[i]] = index
                    count[i] += 1

        columns = []
        for i in range(int(self.num_ToR_port / 2)):
            columns.append("OCS" + str(i))
        self.ToR4OCS_ID_table = pd.DataFrame(ToR4OCS_ID_table, columns=columns, dtype=int)

    # create table to store permutations for every OCS
    def permutation_table_create(self):
        permutation_table = np.zeros([self.num_OCS, self.num_ToR], dtype=int)

        for level in range(self.m_level + 1):
            for i in range(self.num_per_level_OCS * level, self.num_per_level_OCS * (level + 1)):
                for j in range(0, self.num_per_level_OCS, self.k_regular ** (1 - int(self.single_model))):
                    target = self.OCS_ID_table.iloc[(i + j) % self.num_per_level_OCS + self.num_per_level_OCS * level]
                    reference = self.OCS_ID_table.iloc[j]
                    for k in range(self.n_OCS_port):
                        permutation_table[i][reference[k]] = target[k]

        columns = []
        for i in range(self.num_ToR):
            columns.append("ToR" + str(i))
        self.permutation_table = pd.DataFrame(permutation_table, columns=columns, dtype=int)

    # create table to store link pair for every coded_action_id
    def coded_action_table_create(self):
        coded_action_table = np.zeros([int(self.num_actions), 3])  # 3 = 2 port + 1 exchange mode
        coded_action_table[0][0] = 0
        coded_action_table[0][1] = 0
        coded_action_table[0][2] = 2
        count = 1
        for l1 in range(self.num_per_OCS_link):
            for l2 in range(l1 + 1, self.num_per_OCS_link):
                for mode in range(2):
                    coded_action_table[count][0] = l1
                    coded_action_table[count][1] = l2
                    coded_action_table[count][2] = mode
                    count += 1

        columns = []
        for i in range(2):
            columns.append("link" + str(i))
        columns.append("modeID")
        self.coded_action_table = pd.DataFrame(coded_action_table, columns=columns, dtype=int)

    def demand_generate(self, distribution_id):
        standard_demand = np.zeros([self.num_ToR, self.num_ToR])
        normalized_demand = np.zeros([self.num_ToR, self.num_ToR])
        for i in range(self.num_ToR):
            for j in range(i + 1, self.num_ToR):
                if distribution_id == "uniform":
                    normalized_demand[i][j] = random.uniform(0, 1)
                    standard_demand[i][j] = normalized_demand[i][j] * multi_factor
                    # self.standard_demand[j][i] = self.standard_demand[i][j]
                else:
                    r = random.random()
                    l = 1
                    while cdfx[distribution_id][l] < r:
                        l += 1
                        standard_demand[i][j] = sc[distribution_id] * pow(10, (cdfy[distribution_id][l] - cdfy[distribution_id][l - 1]) /
                                                                               (cdfx[distribution_id][l] -
                                                                                cdfx[distribution_id][l - 1]) *
                                                                               (r - cdfx[distribution_id][l - 1]) +
                                                                               cdfy[distribution_id][l - 1])
                        normalized_demand[i][j] = standard_demand[i][j] / multi_factor
                        # self.standard_demand[j][i] = self.standard_demand[i][j]

        return standard_demand, normalized_demand

    # # create demand with already-known flow cdf
    # # we have four cdf : web search, cache, data mining, hadoop
    def demand_load(self, demand_id):
        self.current_demand = self.standard_demand_memory[demand_id][:]
        self.normalized_demand = self.normalized_demand_memory[demand_id][:]

    # # # create demand with already-known flow cdf
    # # # we have four cdf : web search, cache, data mining, hadoop
    # def demand_init(self, distribution_id):
    #     # demand = np.zeros([self.num_ToR, self.num_ToR])
    #     # demand = np.random.randint(0, 100, [self.num_ToR, self.num_ToR])
    #     # demand = np.arange(0, self.num_ToR**2).reshape([self.num_ToR, self.num_ToR])*1e4
    #     if not hasattr(self, 'standard_demand'):
    #         standard_demand, normalized_demand = self.demand_generate(distribution_id)
    #         self.standard_demand = standard_demand
    #     self.current_demand = self.standard_demand
    #     self.normalized_demand = self.standard_demand / multi_factor




