from numpy import *
import operator
def createdataset():
    group=array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2],[18,90]])
    labels=['A','A','A','B','B','B']
    return group,labels
'''
def classify0( inX,dataset,labels,k):
    dataSize=dataset.shape[0]#读取行数
    diffMat=tile(inX,(dataSize,1))-dataset
    sqdiffMat=diffMat**2
    sqdis=sqdiffMat.sum(axis=1)
    distance=sqdis**0.5
    sortIndexDis=distance.argsort()
    classcount={}
    for i in range(k):
        voteIlabel=labels[sortIndexDis[i]]      #labels是标志向量
        classcount[voteIlabel]=classcount.get(voteIlabel,0)+1
    sortclass=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclass[0][0]
    '''
def classify(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort();
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
