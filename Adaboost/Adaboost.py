import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():#加载一个简单的数据集
    dataMat = np.mat([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels     #返回 数据矩阵，标签列表

def stumpClassify(dataMat, dimen, threshVal, threshIneq):#对dataMat基于dimen维度，以thredVal为阈值进行类别划分，threshIneq是与阈值的比较方式
	m,n=np.shape(dataMat)
	retArr=np.ones((m,1))
	if(threshIneq == 'lt'):
		retArr[dataMat[ : ,dimen]<=threshVal]=-1
	else:
		retArr[dataMat[ : ,dimen]>threshVal]=-1
	return retArr   #返回一个列矩阵( m * 1)

def buildStump(dataArr,labelArr,D):#遍历stumpClassify()所有可能输入，找到数据集上的最佳单层决策树
    dataMat=np.mat(dataArr)
    labelMat=(np.mat(labelArr)).T
    m,n=np.shape(dataMat)
    numSteps=10.0
    bestStump={}
    bestClasEst=np.mat(np.zeros( (m,1) ))
    Minerror=np.inf # error is > 999999999999999999
    for i in range (n):     #在所有特征上遍历
        rangeMin=dataMat[ : ,i].min()
        rangeMax=dataMat[ : ,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):         #在该特征的所有值上遍历
            for inequal in ['lt','gt']:         #在大于，小于上切换
                thredVal=rangeMin+float(j)*stepSize
                predictedVal=stumpClassify(dataMat,i,thredVal,inequal)
                errMat=np.mat(np.ones((m,1)))
                errMat[predictedVal==labelMat]=0    #0 is right , 1 is wrong
                weightedError=D.T * errMat    #D是每个样本对应的权重，D m*1，D.T 1*m
                #print("the weighted error is",weightedError,"dim is %d,thresh is %.2f,ineq is %s"%(i,thredVal,inequal))
                if weightedError<Minerror:
                    Minerror=weightedError
                    bestClasEst=predictedVal.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=thredVal
                    bestStump['ineq']=inequal
    #bestStump为划分方式，Minerror是含权重的错误率，bestClasEst预测分类值
    return bestStump,Minerror ,bestClasEst
    
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakArr=[]
    m,n=np.shape(dataArr)
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels,D)
        error=float(error)
        flqalpha=float(0.5*np.log((1-error)/max(error,1e-16)))#每个分类器的权重值,防止÷0溢出
        #print("flqalpha:",flqalpha)
        bestStump['alpha']=flqalpha
        weakArr.append(bestStump)
        #更新下一次迭代时的样本权重值
        preval=np.multiply(np.mat(classLabels).T , classEst)  #multiply (1*m ,1*m ) 得 1 * m,right is 1,error is -1
        expon=-1 * flqalpha * preval  #  1*m
        D=np.multiply(D, np.exp(expon))
        D=D/sum(D)
        # print("下一次循环的样本权重D:",D)
        aggClassEst+=flqalpha*classEst      #数 * (m*1)矩阵
        #print("aggClassEst:",aggClassEst.T)
        errmatIndex=np.sign(aggClassEst) != np.mat(classLabels).T
        aggerr=np.multiply(errmatIndex, np.ones((m,1)) )
        errorRate=aggerr.sum()/m
        #print("errorRate is",errorRate)
        if errorRate==0:
            break
    return weakArr,aggClassEst

def adaClassify(dataToClass,classifierArr):
    dataMatrix=np.mat(dataToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(filename):
    fr=open(filename)
    numfea=len(fr.readline().split('\t'))
    fr.seek(0,0)
    datamat,labelmat=[],[]
    for line in fr.readlines():
        lineArr=[]
        curline=line.strip().split('\t')
        for i in range(numfea-1):
            lineArr.append(float(curline[i]))
        datamat.append(lineArr)
        labelmat.append(float(curline[-1]))
    return datamat,labelmat

def check():
    d,l=loadDataSet('horseColicTraining2.txt')
    classf,agg=adaBoostTrainDS(d,l,10)
    tb,tl=loadDataSet('horseColicTest2.txt')
    p10=adaClassify(tb,classf)
    err=np.mat(np.ones((67,1)))
    errr=err[p10!=np.mat(tl).T].sum()
    rate=errr/67
    plotROC(agg.T,l)
    #print(rate)
'''
ROC receiver operating charateristics 用于非均衡问题的分类评价方法
x=FP/(FP+TN)       y=TP/(TP+FN)
'''
def plotROC(predStrengths,classlabels):
    cur=(1.0,1.0)           # curse position
    ySum=0.0
    numPosClass=sum(np.array(classlabels)==1.0)
    yStep=1/float(numPosClass)
    xStep=1/float(len(classlabels)-numPosClass)
    sortedIndicies=predStrengths.argsort()              #  返回从小到大的索引号
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:  
        if(classlabels[index]==1.0):
            delx,dely=0,yStep
        else:
            delx,dely=xStep,0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delx],[cur[1],cur[1]-dely],c='b')
        cur=(cur[0]-delx,cur[1]-dely)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("false positive rate")
    plt.ylabel("True positive rate")
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])      #plt.axis([a, b, c, d]) :  x轴起始于a，终止于b ，y轴起始于c，终止于d
    plt.show()
    print("the area under curve is:",ySum*xStep)
