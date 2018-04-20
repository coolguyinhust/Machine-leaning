import numpy as np
import matplotlib.pyplot as plt
    
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

def standRegres(xArr,yArr):
    xMat,yMat=np.mat(xArr),np.mat(yArr).T
    xTx=xMat.T * xMat  #n*n
    if np.linalg.det(xTx)==0.0:
        print("This matrix is singular,connot do inverse")
        return
    ws=xTx.I *(xMat.T *yMat)
    return ws               #n*1

def picture(xArr,yArr,ws):
    # xMat,yMat=loadDataSet('ex0.txt')
    #ws=standRegres(xMat,yMat)
    xMat,yMat=np.mat(xArr),np.mat(yArr)
    yHat0=xMat*ws
    print(np.corrcoef(yHat0.T,yMat))        #输出预测值与真实值的相关性
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])   #scatter函数用法？？？？
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show

def lwlr(testPoint,xArr,yArr,k=1.0 ):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat( np.eye( (m) ) )
    for j in range(m):
        diffMat=testPoint -xMat[j,:]
        weights[j,j]=np.exp(diffMat *diffMat.T/(-2.0*k**2))
    xTx=xMat.T*weights*xMat
    if np.linalg.det(xTx)==0:
        print("This matrix is singular,connot do inverse")
        return
    ws=xTx.I *xMat.T * weights*yMat 
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def running_1():
    xArr,yArr=loadDataSet('ex0.txt')
    print(lwlr(xArr[0],xArr,yArr,1.0))
    print(lwlrTest( xArr,xArr,yArr,0.05))

def wei_pic():
    xArr,yArr=loadDataSet('ex0.txt')
    yHat=lwlrTest( xArr,xArr,yArr,1.0)
    xMat,yMat=np.mat(xArr),np.mat(yArr)
    srt=xMat[:,1].argsort(0)
    xsort=xMat[srt][: , 0 , :]
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xsort[:,1],yHat[srt])
    ax.scatter(xMat[:,1].flatten().A[0], yArr,c='red')   #scatter函数用法？？？？
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def running_2():
    abx,aby=loadDataSet('abalone.txt')
    yHat01=lwlrTest(abx[0:99],abx[0:99],aby[0:99],0.1)
    yHat1=lwlrTest(abx[0:99],abx[0:99],aby[0:99],1)
    yHat10=lwlrTest(abx[0:99],abx[0:99],aby[0:99],10)
    a=rssError(aby[0:99],yHat01.T)
    b=rssError(aby[0:99],yHat1.T)
    c=rssError(aby[0:99],yHat10.T)
    print("对自身的训练数据进行自拟合")
    print(a,b,c)
    yHatnew01=lwlrTest(abx[100:199],abx[0:99],aby[0:99],0.105)
    yHatnew1=lwlrTest(abx[100:199],abx[0:99],aby[0:99],1)
    yHatnew10=lwlrTest(abx[100:199],abx[0:99],aby[0:99],10)
    a=rssError(aby[100:199],yHatnew01.T)
    b=rssError(aby[100:199],yHatnew1.T)
    c=rssError(aby[100:199],yHatnew10.T)
    print("对自身的训练数据进行自拟合")
    print(a,b,c)

def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom)==0:
        print("This matrix is singular,connot do inverse")
        return
    ws=denom.I *xMat.T * yMat 
    return ws

def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m,n =np.shape(xMat)
    yMean=np.mean(yMat,axis=0)  #return the average of each row  1*n
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,n))
    linepic=np.zeros((30,1))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
        linepic[i,:]=i-10
    return wMat,linepic

def running_3():
    abx,aby=loadDataSet('abalone.txt')
    linepic,Weights=ridgeTest(abx,aby)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(Weights,linepic)
    plt.show()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m,n =np.shape(xMat)
    yMean=np.mean(yMat,axis=0)  #return the average of each row  1*n
    yMat=yMat-yMean
    xVar=np.var(xMat,0)
    xMean=np.mean(xMat,0)
    xMat=(xMat-xMean)/xVar
    m,n =np.shape(xMat)
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        lowestError=np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssErr=rssError(yMat.A,yTest.A)
                if rssErr<lowestError:
                    lowestError=rssErr
                    wsMat=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def stageWise2(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m,n = np.shape(xMat)
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xVar = np.var(xMat, axis=0) #variance of x
    xMean = np.mean(xMat, axis=0)
    xMat=(xMat - xMean)/xVar
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssErr = rssError(yMat.A, yTest.A)
                if rssErr < lowestError:
                    lowestError = rssErr
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def run4():
    x,y=loadDataSet('abalone.txt')
    res=stageWise(x,y,0.001,numIt=1000)
    print(res)
    print('\n')
    x,y=loadDataSet('abalone.txt')
    a=stageWise2(x,y,0.001,numIt=1000)
    print(a)

if '_name_'=="main":
    run4()
