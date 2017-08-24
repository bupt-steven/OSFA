#!/usr/bin/python
# -*- coding: utf-8 -*
# 目前根据端口先默认fatTree,拓扑结果为一个环
import copy
import random
import numpy as np
from utility import util
from var_dump import var_dump

BANDWIDTH = 1250
SPEED = 15
RTT = 0.1
SSRENTSH = 512
maxm = 16

result = []  # 最短路径输出结果
stack = []  # 最短路径辅助栈
TcpWindow = []

def normalTopo(topo) :
    topo_b = copy.deepcopy(topo)
    for i in range(maxm) :
        for j in range (maxm) :
            if topo_b[i][j] > 0 :
                topo_b[i][j] = 1
    return topo_b
# Dijkstra求最短路，求出一个包含最短的前驱节点的数组
def Dijkstra(topoNormal, v=0):
    maxInt = 1000000
    dist = []  # 记录最短距离
    pre = []  # 纪录上一节点
    queue = []  # 队列
    # 初始化不可到达点
    for i in range(len(topoNormal)):
        for j in range(len(topoNormal)):
            if j != i and topoNormal[i][j] == 0:
                topoNormal[i][j] = maxInt
    for i in range(len(topoNormal)):
        templist = []
        dist.append(topoNormal[v][i])
        queue.append(False)
        if dist[i] == maxInt:
            templist.append(-1)
            pre.append(templist)
        else:
            templist.append(v)
            pre.append(templist)
    dist[v] = 0
    queue[v] = True
    for i in range(1, len(topoNormal)):
        mindist = maxInt  # 当前最小值
        u = v  # 前驱节点
        for j in range(len(topoNormal)):
            if queue[j] == False and dist[j] < mindist:
                u = j
                mindist = dist[j]
        queue[u] = True
        for j in range(len(topoNormal)):
            if queue[j] == False and topoNormal[u][j] < maxInt:
                if dist[u] + topoNormal[u][j] < dist[j]:  # 在通过新加入的u点路径找到离v0点更短的路径
                    dist[j] = dist[u] + topoNormal[u][j]  # 更新dist
                    pre[j][0] = u  # 纪录前驱顶点
                if dist[u] + topoNormal[u][j] == dist[j]:  # 在通过新加入的u点路径找到离v0点更短的路径
                    if pre[j][0] == -1:
                        pre[j][0] = u
                    elif u not in pre[j]:
                        pre[j].append(u)
    return pre


# 解析前驱数组，求出最短路径
def dfs(pre, targetNode, sourceNode):
    # return 一个数组，每一列纪录一条最短路径
    global stack
    global result
    for i in range(len(pre[targetNode])):
        stack.append(pre[targetNode][i])
        if pre[targetNode][i] == sourceNode:
            result.append(copy.deepcopy(stack))
            stack.pop()
            return
        else:
            dfs(pre, pre[targetNode][i], sourceNode)
        stack.pop()


# 将每条边分配一个id
def edgeHash(topoNormal):
    outPut = []
    for i in range(len(topoNormal)):
        for j in range(len(topoNormal)):
            if topoNormal[i][j] == 1:
                temp = str(i) + "," + str(j)
                outPut.append(temp)
    return outPut


# 将任务划分成流，其中选用了ECMP的规则
def taskToFlow(task, topoNormal):
    global result
    shortRouter = []
    for i in range(len(task)):
        result = []
        fromTo = task[i][0].split(",")
        preArray = Dijkstra(topoNormal, int(fromTo[0]))
        for j in range(len(preArray)) :
            if -1 in preArray[j] :
                return shortRouter, True
        dfs(preArray, int(fromTo[1]), int(fromTo[0]))
        shortRouter.append(result)
        for t in range(len(shortRouter[i])):
            shortRouter[i][t].insert(0, int(fromTo[1]))
    return shortRouter,False

# 将任务划分成流，其中选用了ECMP的规则
def taskToFlow1(task, topoNormal):
    global result
    shortRouter = []
    for i in range(len(task)):
        result = []
        fromTo = task[i][0].split(",")
        preArray = Dijkstra(topoNormal, int(fromTo[0]))
        dfs(preArray, int(fromTo[1]), int(fromTo[0]))
        shortRouter.append(result)
        for t in range(len(shortRouter[i])):
            shortRouter[i][t].insert(0, int(fromTo[1]))
    return shortRouter

# 分配流ID
def getFlowId(shortRouter):
    flowInfo = []
    for num in range(len(shortRouter)):
        for i in range(len(shortRouter[num])):
            flowInfo.append(shortRouter[num][i])
    return flowInfo


# 分配每条流的大小
def getFlowNum(shortRouter, task):
    flowNumInfo = []
    for num in range(len(shortRouter)):
        for i in range(len(shortRouter[num])):
            flowNumInfo.append(task[num][1] / len(shortRouter[num]))
    return flowNumInfo


# 将流分配到每条边上
def flowToEdge(flowInfo, edgeHashInfo, flowNumInfo):
    edgeFlowInfo = {}
    for key in range(len(flowInfo)):
        for i in range(len(flowInfo[key]) - 1):
            # if flowInfo[key][i] > flowInfo[key][i + 1]:
            tempData = str(flowInfo[key][i]) + "," + str(flowInfo[key][i + 1])
            # else:
            #     tempData = str(flowInfo[key][i + 1]) + "," + str(flowInfo[key][i])
            if tempData in edgeHashInfo:
                k = edgeHashInfo.index(tempData)
                if k in edgeFlowInfo:
                    edgeFlowInfo[k] = edgeFlowInfo[k] + "," + str(key)
                else:
                    edgeFlowInfo[k] = str(key)
    return edgeFlowInfo

def intitialTcpWindow(flowNumInfo,TcpFlag,TcpWindow) :
    # flowNumInfo的key为每条流的id
    # 正式系统中每次初始化后启动窗口值为随机值，现在手动规定的，方便观察。
    if TcpFlag == 0 :
        TcpWindow = []
        for i in range(len(flowNumInfo)):
            TcpWindow.append(1)
    else :
        for j in range(len(flowNumInfo)):
            if j >= len(TcpWindow) :
                TcpWindow.append(TcpWindow[j%len(TcpWindow)])
    return TcpWindow
def step_time(change_interval ,edgeFlowInfo, flowNumInfo,edgeHashInfo, topo,TcpWindow):
    T = change_interval
    # 下面开始主逻辑，循环
    for curtime in range(T):
        # 先将其增大，在判断是否拥塞，决定是否减半
        for i in range(len(TcpWindow)):
            if TcpWindow[i] < SSRENTSH:  # 暂时将sstrensh 值定位8
                TcpWindow[i] *= 2
            else:
                TcpWindow[i] += 1
        for num,numVal in edgeFlowInfo.items():
            tranSum = 100000000000  # 保证进循环
            salt = 0
            topoBandWidthArr = edgeHashInfo[num].split(",")
            topoBandWid = topo[int(topoBandWidthArr[0])][int(topoBandWidthArr[1])]
            while tranSum > BANDWIDTH :  # 暂时每条边的最大传输值为 20
                flowArray = edgeFlowInfo[num].split(",")  # 将这条边的flowid上的所有流分割出来
                # print(num)
                tranSum = 0
                for j in range(len(flowArray)):
                    tranSum += int(TcpWindow[int(flowArray[j])] * SPEED)  # 计算目前情况下这条边上所有流大小
                if tranSum <= BANDWIDTH * topoBandWid:
                    break
                else:
                    salt += 1
                    decreaseFlow = ((curtime+num+j+1+salt) *  24036583) % len(flowArray)
                    TcpWindow[int(flowArray[decreaseFlow])] /= 2
                    # print(decreaseFlow, "你被减速了")
            # print(tranSum,"-----",BANDWIDTH * topoBandWid, "        ",curtime)
        # print("--------------------------------------start-------------------------------------------")
        # print(TcpWindow)
        # print("---------------------------------------end--------------------------------------------")
        #此时已经计算完了无拥塞下的各个TCP的窗口

        for idx in range(len(flowNumInfo)):
            if flowNumInfo[idx] > (TcpWindow[idx] * SPEED * RTT):
                flowNumInfo[idx] -= (TcpWindow[idx] * SPEED * RTT)
            else:  # 该条流传完了
                flowNumInfo[idx] = 0
                keys = []
                for idkey,v in edgeFlowInfo.items():
                    keys.append(idkey)
                # print("*****************************************************************")
                # print(edgeFlowInfo)
                # print(keys)
                # print("******************************************************************")
                for key in range(len(keys)):
                    clearArray = edgeFlowInfo[keys[key]].split(",")
                    idxStr = str(idx)
                    if idxStr in clearArray:
                        needDelete = clearArray.index(idxStr)
                        # print(clearArray[needDelete], "你被传完了，时间是", curtime, "剔除的边是", key)
                        del clearArray[needDelete]
                        if len(clearArray) == 0:
                            del edgeFlowInfo[keys[key]]
                        else:
                            clearArrayStr = ",".join(clearArray)
                            edgeFlowInfo[keys[key]] = clearArrayStr
            # print(flowNumInfo)
            flowSum = 0
            for i in range(len(flowNumInfo)) :
                flowSum += flowNumInfo[i]
            if flowSum == 0 :
                return curtime+1,flowNumInfo
        # print(flowNumInfo)
    return change_interval, flowNumInfo
def OutPutDemand (flowInfo,demandRest):
    #组装demand begin
    demandNext = np.zeros([maxm,maxm])
    for k in range(len(demandRest)) :
        if demandRest[k] != 0 :
            # if flowInfo[k][0] > flowInfo[k][-1] :
            #     demandNext[flowInfo[k][-1]][flowInfo[k][0]] += demandRest[k]
            # else :
            demandNext[flowInfo[k][-1]][flowInfo[k][0]] += demandRest[k]
            # print(flowInfo[k][0],flowInfo[k][-1],demandRest[k])
    return demandNext
    #组装demand end
def demandToTask (demand) :
    task = []
    for i in range (maxm) :
        for j in range (maxm) :
            if demand[i][j] != 0 :
                temp = [str(i)+","+str(j),demand[i][j]]
                task.append(temp)
    return task
# if __name__ == '__main__':
#     file_path = "../../seed_3/s_demand0.txt"



def streamMain(timeLength,topo, demand, TcpFlag = 0) : #TcpFlag 1 从全局变量读取，0初始化为1
    #pre = Dijkstra(topoNormal, 0)
    topoNormal = normalTopo(topo)

    result = []  # 最短路径输出结果
    stack = []  # 最短路径辅助栈

    egdeHashInfo = []
    # 边信息，每条边有一个唯一id，值为字符串（1，2），代表1连接2
    taskInfo = []
    # 每个任务，key值为taskid，value一个数组，其中value［0］如（0，4），起始节点，value［1］为task的流大小
    shortRouter = []
    # 最短路径记录。key值为taskid，value为一个数组，每一值代表一个路径；
    flowInfo = []
    # 将flow映射到每条边上，key值为flowid，value为egdeid
    edgeFlowInfo = []
    # 记录每条边需要传的flowid，以及大小
    edgeTcpLimitInfo = []
    # 记录每条边的Tcp阀值
    tm = 1
    # 默认TCP的一跳时间

    #初始化任务
    task = demandToTask(demand)

    shortRouter, unArrivalFlag = taskToFlow(task, topoNormal)
    if (unArrivalFlag == True) :
        return 0,demand,demand, True
    flowInfo = getFlowId(shortRouter)

    flowNumInfo = getFlowNum(shortRouter, task)
    # var_dump(len(flowNumInfo))
    edgeHashInfo = edgeHash(topoNormal)
    #var_dump(edgeHashInfo)
    edgeFlowInfo = flowToEdge(flowInfo, edgeHashInfo, flowNumInfo)
    global TcpWindow
    TcpWindow = intitialTcpWindow(flowNumInfo,TcpFlag,TcpWindow)
    spendTime, demandRest = step_time(timeLength,edgeFlowInfo,flowNumInfo,edgeHashInfo,topo,TcpWindow) #获得时间片后的情况
    # print(demandRest)
    demandNext = OutPutDemand(flowInfo,demandRest)
    resultDemand = np.array(demandNext)
    demandNP = np.array(demand)
    #print(resultDemand)
    # print(TcpWindow)
    return spendTime,resultDemand,demandNP,False

# 不检测	
def streamMain1(timeLength,topo, demand, TcpFlag = 0) : #TcpFlag 1 从全局变量读取，0初始化为1
    #pre = Dijkstra(topoNormal, 0)
    topoNormal = normalTopo(topo)

    result = []  # 最短路径输出结果
    stack = []  # 最短路径辅助栈

    egdeHashInfo = []
    # 边信息，每条边有一个唯一id，值为字符串（1，2），代表1连接2
    taskInfo = []
    # 每个任务，key值为taskid，value一个数组，其中value［0］如（0，4），起始节点，value［1］为task的流大小
    shortRouter = []
    # 最短路径记录。key值为taskid，value为一个数组，每一值代表一个路径；
    flowInfo = []
    # 将flow映射到每条边上，key值为flowid，value为egdeid
    edgeFlowInfo = []
    # 记录每条边需要传的flowid，以及大小
    edgeTcpLimitInfo = []
    # 记录每条边的Tcp阀值
    tm = 1
    # 默认TCP的一跳时间

    #初始化任务
    task = demandToTask(demand)

    shortRouter= taskToFlow1(task, topoNormal)
    # if (unArrivalFlag == True) :
    #     return 0,demand,demand, True
    flowInfo = getFlowId(shortRouter)

    flowNumInfo = getFlowNum(shortRouter, task)
    # var_dump(len(flowNumInfo))
    edgeHashInfo = edgeHash(topoNormal)
    #var_dump(edgeHashInfo)
    edgeFlowInfo = flowToEdge(flowInfo, edgeHashInfo, flowNumInfo)
    global TcpWindow
    TcpWindow = intitialTcpWindow(flowNumInfo,TcpFlag,TcpWindow)
    spendTime, demandRest = step_time(timeLength,edgeFlowInfo,flowNumInfo,edgeHashInfo,topo,TcpWindow) #获得时间片后的情况
    # print(demandRest)
    demandNext = OutPutDemand(flowInfo,demandRest)
    resultDemand = np.array(demandNext)
    demandNP = np.array(demand)
    #print(resultDemand)
    # print(TcpWindow)
    return spendTime,resultDemand,demandNP

# demand = util.readfile(file_path)
# topo = util.fatTreeInit(6)
# hh = 45
# # topo = util.BuildDCells(4,2)
# topoNormal = normalTopo(topo)
# demandnn = np.zeros([hh,hh])
# for i in range(hh) :
#     for j in range(hh) :
#         if i < 16 and j < 16:
#             demandnn[i][j] = demand[i][j]
#         else :
#             demandnn[i][j] = 0
#
# # demand = np.zeros([maxm,maxm])
# # for i in range(maxm) :
# #     for j in range(maxm) :
# #         demand[i][j] = random.randrange(1000, 1250)
#         # demand[i][j] = 12
# # demand[0][5] = 125
# # spendTime, demandNext, demandCur,Flag = streamMain(8000,topo, demandnn, 0)
# spendTime, demandNext, demandCur = streamMain1(8000,topo, demandnn, 0)
# print(spendTime)
# print(demandNext)
# print(demandCur)
# print(Flag)

