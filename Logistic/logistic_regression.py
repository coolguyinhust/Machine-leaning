import matplotlib.pyplot as plt
from numpy import *

def sigmoid(Z):     #Z可以是numpy.array, numpy.mat
    return 1.0/(1+exp(-Z))

def loadDataSet():
    dataMat,labelMat=[],[]
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            arr=line.strip().split()
            dataMat.append([1.0,float(arr[0]),float(arr[1])])
            labelMat.append(int(arr[2]))
        return dataMat ,labelMat

def gradAscent(dataMat,classLabels):#梯度上升法每一次都用到了所有样本值
    dataMatrix=mat(dataMat)
    labelMat=mat(classLabels)
    m,n=shape(dataMatrix)   #m is the number of test data, n is the chararistics of a data element
    alpha=0.001
    maxCycles=500
    weight=mat(ones((1,n)))#1行 n列
    for k in range(maxCycles):
        hMat=sigmoid(dataMatrix*weight.T)  #    m * n       n * 1     sigmoid函数的参数此处是一个m*1的矩阵(列向量)，每行只有一个元素Zi，Zi=Xi1*Wi1+Xi2*Wi2+...+Xin*Win
        error=labelMat-hMat.T   #1,m
        grad=error*dataMatrix #1,m
        deta=alpha*error*dataMatrix#1,n
        weight=weight+deta
    return weight.getA().ravel()#getA()用于Matrix to nparray,ravel()用于降维

def plotbestfit(weight):
    dataMat,labelMat=loadDataSet()
    dataarr=array(dataMat)
    xcord1,ycord1=[],[]
    xcord0,ycord0=[],[]
    m=len(labelMat)
    for i in range(m):
        if int(labelMat[i]):
            xcord1.append(dataarr[i,1])
            ycord1.append(dataarr[i,2])
        else:
            xcord0.append(dataarr[i,1])
            ycord0.append(dataarr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord0,ycord0,s=30,c='blue')
    x=arange(-3.0,3.0,0.1)
    y=(-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()  #running plotbestfit(array([ 4.12414349,  0.48007329, -0.6168482 ]))


def stocGradAscent0(dataMatrix,classLabels):#随机梯度上升法只用一个样本点更新回归系数
    m,n=shape(dataMatrix)
    weight=ones(n) #1*n
    alpha=0.01
    for i in range(m):
        z=sum(dataMatrix[i]*weight.T) #  n * 1  1 * n
        h=sigmoid(z)
        error=classLabels[i]-h
        weight+= alpha*error*dataMatrix[i]
    return weight

def stocGradAscent1(dataMatrix,classLabels,numIter=50):
    m,n=shape(dataMatrix)
    weight=ones(n) #1*n
    for j in range(numIter):#循环numIter次
        dataIndex=list (range(m))
        for i in range(m):#第i个选出来的样本是编号为randIndex的样本（随机选取样本）
            alpha=4/(1+i+j)+0.01      #步长为何这样设计？随着遍历的次数越来越多，步长越小，满足参数收敛
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weight.T))
            error=classLabels[randIndex]-h
            weight+= alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weight

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet,trainingLabels=[],[]
    for line in frTrain.readlines():
        curr=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(curr[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curr[21]))
    trainweights=stocGradAscent1(array(trainingSet),trainingLabels,1000)
    errorCount=0
    numTestvec=0
    testSet, testLabels = [], []
    for line in frTest.readlines():
        numTestvec+=1
        curr2=line.strip().split('\t')
        lineArr2=[]
        for i in range(21):
    	    lineArr2.append(float(curr2[i]))
        testSet.append(lineArr2)
        testLabels.append(float(curr2[21]))
    prob=sigmoid(mat(testSet)*mat(trainweights).T).T#m *n  *  (1*n).T  即m*1 。prob是一个(m*1).T  即1*m的矩阵
    prob=array(prob)
    for i in range(numTestvec):
        prob[0][i]=int(prob[0][i]>0.5)
        if(prob[0][i] != testLabels[i]):
            errorCount+=1
    errorRate=float(errorCount/numTestvec)
    print("the error rate of this test is :%f "%errorRate)
    return errorRate

'''
    numpy.mat()说明：
   Q: [[1][2][3]]和[[1,2,3]]代表什么？
   A: 3行1列和1行3列
'''
def multiTest():
    numTest=10;errorSum=0.0
    for k in range(numTest):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is :%f"%(numTest,errorSum/float(numTest)))
