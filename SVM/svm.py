from numpy import *
def smoSimple(dataMatin,classLabels,C,toler,maxIter):
	dataMatin,labelMat=mat(dataMatin),mat(classLabels).transpose()
	b=0;m,n =shape(dataMatin)
	alphas=mat(zeros(m,1))
	iter=0
	while(iter<maxIter):
		alphaPairsChanged=0
	return b,alphas
