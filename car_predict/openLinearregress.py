#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Created on OCT09, 2015
Linear Regression algorithm
@author: Soledede
'''
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('carpredict.pdf')
from pylab import *
import numpy as np
from sklearn import  linear_model 
import sys

prePutNum = -1
preBidNum = -1

prePutNum = raw_input("请输入投标数量: ")
preBidNum = raw_input("请输入竞标人数: ")

#print prePutNum
#print preBidNum
'''
for i in range(1, len(sys.argv)):
    prePutNum sys.argv[i]
'''
# Build X, Y from 2nd file
f = open('/home/spark/wbj/lr/linear_regression/car_predict.txt')
lines = f.readlines()
auctionTime = []
putNum = []
price = []
bidNum = []

for line in lines:
    line = line.replace("\n", "")
    vals = line.split("\t")
    auctionTime.append(float(vals[1]))
    putNum.append(float(vals[2]))
    price.append(float(vals[0]))
    bidNum.append(float(vals[3]))
auctionTime = np.array(auctionTime)
#print auctionTime
putNum = np.array(putNum)
price = np.array(price)
bidNum = np.array(bidNum)

# linregress doesn't do multi regression, so we use sklearn
#ones = np.ones(auctionTime.shape)
x = np.vstack([putNum, bidNum]).T # don't need ones for 
regr = linear_model.LinearRegression()
#print 'x:',x
regr.fit( x, price )
predict = "火星价格"
if(prePutNum!=-1):
	predict = regr.predict([float(prePutNum), float(preBidNum)]) 

print "预测的竞标价格：",predict
#predict = regr.predict(3000) 
#print predict
#print regr.coef_

###########################################################
x = np.vstack(auctionTime)
#print 'x:',x
regr.fit( x, price )
#predict = regr.predict([3000, 3]) 
#predict = regr.predict(3000) 
#print predict
#print regr.coef_

figure(1)
subplot(221)  
plt.scatter(x, price, color='red')
plt.plot(x, regr.predict(x), color='blue')

plt.xticks(())
plt.yticks(())
plt.axis('tight')
plt.xlabel('auctionTime')
plt.ylabel('price')

################putNum
x = np.vstack(putNum)
regr.fit( x, price )
subplot(222)  
plt.scatter(x, price, color='red')
plt.plot(x, regr.predict(x), color='green')

plt.xticks(())
plt.yticks(())
plt.axis('tight')
plt.xlabel('put quantity')
plt.ylabel('price')

####################################bidNum
x = np.vstack(bidNum)
regr.fit( x, price )
subplot(212)  
plt.scatter(x, price, color='red')
plt.plot(x, regr.predict(x), color='black')

plt.xticks(())
plt.yticks(())
plt.axis('tight')
plt.xlabel('bid number')
plt.ylabel('price')


#plt.show()
plt.savefig(pp, format='pdf')
pp.close()
