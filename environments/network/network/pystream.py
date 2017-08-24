#!/usr/bin/python
# -*- coding: utf-8 -*
# 目前根据端口先默认fatTree,拓扑结果为一个环
import copy  
import random
# 从文件中读topo结构
def initial(lineNum = 8) :
    file_object = open('topo.txt')
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    line = all_the_text.split("\n")
    element = []
    for num in range(len(line)-1) :
        temp = line[num].split()
        tempInt = [] 
        for index in range(len(temp)) :
             tempInt.append(int(temp[index]))
        element.append(tempInt)
    return element
# Dijkstra求最短路，求出一个包含最短的前驱节点的数组
def Dijkstra(topo, v=0) :
    maxInt = 10000
    dist  = [] #记录最短距离
    pre   = [] #纪录上一节点
    queue = [] #队列
    #初始化不可到达点
    for i in range(len(topo)) :
        for j in range(len(topo)) :
            if j != i and topo[i][j] == 0 :
                topo[i][j] =  maxInt
    for i in range(len(topo)) :
        templist = []
        dist.append(topo[v][i])
        queue.append(False)
        if dist[i] == maxInt :
            templist.append(-1)
            pre.append(templist)
        else :
           templist.append(v)
           pre.append(templist)
    dist[v] = 0
    queue[v] = True
    for i in range(1,len(topo)) :
        mindist = maxInt #当前最小值
        u = v #前驱节点
        for j in range (len(topo)) :
            if queue[j] == False and dist[j] < mindist :
                u = j
                mindist = dist[j]
        queue[u] = True
        for j in range (len(topo)) :
            if queue[j] == False  and topo[u][j] < maxInt :
                if dist[u] + topo[u][j] < dist[j] :  #在通过新加入的u点路径找到离v0点更短的路径
                    dist[j] = dist[u] + topo[u][j]  #更新dist
                    pre[j][0] = u #纪录前驱顶点
                if dist[u] + topo[u][j] == dist[j] :  #在通过新加入的u点路径找到离v0点更短的路径
                    if pre[j][0] == -1 :
                        pre[j][0] = u
                    elif u not in pre[j]:
                        pre[j].append(u)
    return pre

#解析前驱数组，求出最短路径
def dfs(pre , targetNode, sourceNode) :
    #return 一个数组，每一列纪录一条最短路径 
    global stack
    global result
    for i in range(len(pre[targetNode])) :
       stack.append(pre[targetNode][i])
       if pre[targetNode][i] == sourceNode :
           result.append(copy.deepcopy(stack))
           stack.pop()
           return 
       else :
           dfs(pre, pre[targetNode][i], sourceNode)
       stack.pop()

# 将每天边分配一个id
def edgeHash (topo) :
    outPut = []
    for i in range (len(topo)) :
        for j in range (i) :
            if topo[i][j] == 1 :
                temp = str(i) + "," + str(j)
                outPut.append(temp)
    return outPut
#将任务划分成流，其中选用了ECMP的规则
def taskToFlow(task ,topo) :
    global result;
    shortRouter = []
    for i in range(len(task)) :
        result = []
        fromTo = task[i][0].split(",")
        preArray = Dijkstra(topo, int(fromTo[0]))
        dfs(preArray, int(fromTo[1]), int(fromTo[0]))
        shortRouter.append(result)
        for t in range(len(shortRouter[i])) :
            shortRouter[i][t].insert(0,int(fromTo[1]))
    return shortRouter
#分配流ID
def getFlowId( shortRouter) :
    flowInfo = []
    for num in range (len(shortRouter)) :
       for i in range (len(shortRouter[num])) :
           flowInfo.append(shortRouter[num][i])
    return flowInfo
#分配每条流的大小
def getFlowNum(shortRouter, task) :
    flowNumInfo = []
    for num in range (len(shortRouter)) :
       for i in range (len(shortRouter[num])) :
           flowNumInfo.append(task[num][1]/len(shortRouter[num]))
    return flowNumInfo
#将流分配到每条边上
def flowToEdge(flowInfo, edgeHashInfo, flowNumInfo) :
    edgeFlowInfo = {}
    for key in range (len(flowInfo)) :
        for i in range (len(flowInfo[key])-1) :
            if flowInfo[key][i] >  flowInfo[key][i+1] :
                tempData = str(flowInfo[key][i]) + "," + str(flowInfo[key][i+1])
            else :
                tempData = str(flowInfo[key][i+1]) + "," + str(flowInfo[key][i])
            if tempData in edgeHashInfo :
                k = edgeHashInfo.index(tempData)
                if edgeFlowInfo.has_key(k) :
                    edgeFlowInfo[k] = edgeFlowInfo[k] + "," + str(key) 
                else :
                    edgeFlowInfo[k]  = str(key)
    return edgeFlowInfo
     
def intitialTcpWindow(flowNumInfo) :
     #flowNumInfo的key为每条流的id
     #正式系统中每次初始化后启动窗口值为随机值，现在手动规定的，方便观察。
     outPut = []
     for i in range (len(flowNumInfo)) :
         outPut.append(i+1) # 暂时每个加一，让每个窗口不一样
     return outPut


topo = initial(8)
pre = Dijkstra(topo,0)

result = [] #最短路径输出结果
stack  = [] #最短路径辅助栈
#dfs(pre, 4, 0)


egdeHashInfo = [];
#边信息，每条边有一个唯一id，值为字符串（1，2），代表1连接2
taskInfo = [];
#每个任务，key值为taskid，value一个数组，其中value［0］如（0，4），起始节点，value［1］为task的流大小
shortRouter = [];
#最短路径记录。key值为taskid，value为一个数组，每一值代表一个路径；
flowInfo = [];
#将flow映射到每条边上，key值为flowid，value为egdeid
edgeFlowInfo = [];
#记录每条边需要传的flowid，以及大小
edgeTcpLimitInfo = [];
#记录每条边的Tcp阀值
tm  = 1 ;
#默认TCP的一跳时间
T = 4;
#默认一个时间段T内不会有发生拓扑改变。
taskEle = ["0,4",12]
task = []
task.append(taskEle)
task.append(taskEle)

shortRouter = taskToFlow(task, topo);
flowInfo = getFlowId(shortRouter);
flowNumInfo = getFlowNum(shortRouter,task);
egdeHashInfo = edgeHash(topo);
edgeFlowInfo =  flowToEdge(flowInfo,egdeHashInfo, flowNumInfo);
TcpWindow = intitialTcpWindow(flowNumInfo);
#下面开始主逻辑，循环
for curtime in range(T) :
    #todo 如何获取超时呢
    #先将其增大，在判断是否拥塞，决定是否减半
    for i in range (len(TcpWindow)) :
        if TcpWindow[i] < 8 :#暂时将sstrensh 值定位8 
            TcpWindow[i] = TcpWindow[i]*2
        else :
            TcpWindow[i] = TcpWindow[i] + 1
    for num in range(len(edgeFlowInfo)) :
        tranSum = 100000; #保证进循环
        while ( tranSum > 20) : #暂时每条边的最大传输值为 20
            flowArray = edgeFlowInfo[num].split(",")#将这条边的flowid上的所有流分割出来
            tranSum = 0
            for j in range (len(flowArray)) :
                tranSum += int(TcpWindow[int(flowArray[j])]*1) #计算目前情况下这条边上所有流大小
            if tranSum <= 20 : 
                break
            else :
                decreaseFlow = random.randrange(0, len(flowArray)-1)
                TcpWindow[int(flowArray[decreaseFlow])] = TcpWindow[int(flowArray[decreaseFlow])]/2;
                print decreaseFlow,"你被减速了","\n";
    print "-----------------------------------------------------------------------------------\n"
    print TcpWindow
    print "-----------------------------------------------------------------------------------\n"
    #此时已经计算完了无拥塞下的各个TCP的窗口
   
    for idx in range (len(flowNumInfo)) :
        if flowNumInfo[idx] > (TcpWindow[idx] * 1) :
            flowNumInfo[idx] = flowNumInfo[idx] - (TcpWindow[idx]*1);
        else : # 该条流传完了
            flowNumInfo[idx] = 0
            for key,value  in edgeFlowInfo.items() :
                clearArray = value.split(",")
                idxStr = str(idx)
                if idxStr in clearArray :
                    needDelete = clearArray.index(idxStr)
                    print clearArray[needDelete], "你被传完了，时间是",curtime,"剔除的边是",key,"\n"
                    del clearArray[needDelete]
                    if len(clearArray) == 0 :
                        del edgeFlowInfo[key]
                    else :
                        clearArrayStr = ",".join(clearArray) 
                        edgeFlowInfo[key] = clearArrayStr
        print flowNumInfo

