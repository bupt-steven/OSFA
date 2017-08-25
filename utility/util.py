import numpy as np
from var_dump import var_dump
import matplotlib.pyplot as plt

# read file from filePath
def readfile(file_path):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    line = all_the_text.split("\n")
    element = []
    for num in range(len(line) - 1):
        temp = line[num].split()
        tempInt = []
        for index in range(len(temp)):
            tempInt.append(float(temp[index]))
        element.append(tempInt)
    return element


# write file to filePath
def writefile(data, file_path):
    with open(file_path, "w") as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j])+" ")
            f.write("\n")
        # print("successfully write data to file")


# read file from filePath
def readfile(file_path):
    file_object = open(file_path)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    line = all_the_text.split("\n")
    element = []
    for num in range(len(line) - 1):
        temp = line[num].split()
        tempInt = []
        for index in range(len(temp)):
            tempInt.append(temp[index])
        element.append(tempInt)
    return element


# fatTree topo init
def fatTreeInit(K):  # K is only even
    torNum = int(K * K / 2)
    switchNum = int(K * K * 5 / 4)
    coreNum = int(K * K / 4)
    adj = np.zeros([switchNum, switchNum])
    for i in range(K):
        for j in range(int(K / 2)):
            for k in range(int(K / 2)):
                src = int(i * K / 2 + j)
                dest = int(torNum + i * K / 2 + k)
                adj[src][dest] = 1
                adj[dest][src] = 1
    for i in range(coreNum):
        for j in range(K):
            src = int(i + torNum * 2)
            dest = int(torNum + i / (K / 2) + j * K / 2)
            adj[src][dest] = 1
            adj[dest][src] = 1
    return adj


def SumDcell(n, l):
    # this function return the sum of node in the Dcell
    if l == 1:
        return n
    else:
        return SumDcell(n, l - 1) * (SumDcell(n, l - 1) + 1)


def HashEdge(begin, incr, prefix):
    edgeMap = {}
    for i in range(incr):
        tempStr = prefix + "," + str(i)
        edgeMap[tempStr] = begin + i
    return edgeMap


def BuildDCells(n, l):
    # l is the level of the Dcell
    # n is the num of Dcell0
    nodeNum = SumDcell(n, l)
    nodeAllSum = nodeNum + int(nodeNum / n)  # mini switch
    topoBuild = np.zeros([nodeAllSum, nodeAllSum])
    for k in range(l):
        if k == 0:
            # link mini switch
            for index in range(int(nodeNum / n)):
                for times in range(n):
                    src = int(index + nodeNum)  # mini swtich
                    dest = int(index * n + times)
                    topoBuild[src][dest] = 1
                    topoBuild[dest][src] = 1
            continue
        tl = SumDcell(n, k)  # t1 is virtual node in this Dcell
        realMap = {}
        for q in range(int(nodeNum / (tl))):
            begin = q * tl
            incrm = tl
            prefix = str(int(q % (tl + 1)))
            idhashmap = HashEdge(begin, incrm, prefix)
            gl = tl + 1
            if (q + 1) % gl == 0:
                realMap.update(idhashmap)
                for i in range(tl):
                    for j in range(i + 1, gl):
                        uid1 = str(i) + "," + str(j - 1)
                        uid2 = str(j) + "," + str(i)
                        print(realMap[uid1], "----", realMap[uid2])
                        topoBuild[realMap[uid1]][realMap[uid2]] = 1
                        topoBuild[realMap[uid2]][realMap[uid1]] = 1
                realMap = {}
            else:
                realMap.update(idhashmap)
    return topoBuild
def DeBruijnGraph(degree, dim):
    nodeNum = degree ** dim
    topoOut = np.zeros([nodeNum, nodeNum])
    topoHash = {}
    # ttt = numToNdec(12,4,4)
    for i in range(nodeNum):
        temp = numToNdec(i, degree, dim)
        t = "".join(temp)
        topoHash[t] = i
    for key, value in topoHash.items():
        keyList = list(key)
        for j in range(degree):
            prefix = keyList[1:]
            prefix.append(str(j))
            keyData = "".join(prefix)
            topoOut[topoHash[keyData]][value] = 1
            topoOut[value][topoHash[keyData]] = 1
    return topoOut

    # tt = BuildDCells(2,3)
    # tt = fatTreeInit(4)
    # fig = plt.figure()
    # fig, axarr = plt.subplots(2)
    # ax0 = fig.add_subplot(121)
    # im0 = ax0.imshow(tt)
    # plt.colorbar(im0, fraction=0.046, pad=0.04)
    # plt.show()
    # print(tt)

# if __name__ == '__main__':
#     file_path = "../environments/topo.txt"
#     topo = readfile(file_path)
#     print(topo)
#     to_file_path = "../environments/123.txt"
#     writefile(topo, to_file_path)
