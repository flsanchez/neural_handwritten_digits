import numpy as np
import random
import matplotlib.pyplot as plt

class network(object):

    def __init__(self,sizes):
        self.l = len(sizes)
        self.sizes = sizes
        self.b = [np.random.randn(j,1) for j in sizes[1:]]
        self.w = [np.random.randn(j,k) for j,k in zip(sizes[1:],sizes[:-1])]

    def evalOutput(self,a):
        for b,w in zip(self.b,self.w):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,trainset,eta,m,epochs,testset):
        data = zip(trainset[0],trainset[1])
        test = zip(testset[0],testset[1])
        for epoch in range(epochs):
            random.shuffle(data)
            miniBatches = [data[k:k+m] for k in range(0,len(data),m)]
            for miniBatch in miniBatches:
                self.GDMiniBatch(miniBatch,eta)
            print "Epoch {i}/{j} ({k}/{l})".format(i=epoch+1,j=epochs,k=self.evaluate(test),l=len(test))

    def GDMiniBatch(self,miniBatch,eta):
        grad_w = [np.zeros(self.w[i].shape) for i in range(len(self.w))]
        grad_b = [np.zeros(self.b[i].shape) for i in range(len(self.b))]
        m = len(miniBatch)
        for x,y in miniBatch:
            grad_w_mini, grad_b_mini = self.backprop(x,y)
            grad_w = [w+dw for w,dw in zip(grad_w,grad_w_mini)]
            grad_b = [b+db for b,db in zip(grad_b,grad_b_mini)]
        self.w = [w-(float(eta)/m)*dw for w,dw in zip(self.w,grad_w)]
        self.b = [b-(float(eta)/m)*db for b,db in zip(self.b,grad_b)]

    def backprop(self,x,y):
        a = [x]
        z = [0]
        L = self.l
        w = self.w
        b = self.b
        for l in range(0,L-1):
            z.append(np.dot(w[l],a[l])+b[l])
            a.append(sigmoid(z[-1]))
        # print len(a)
        # for i in a: print i.shape
        delta = [np.zeros(a[i].shape) for i in range(L)]
        delta[-1] = self.gradCost(y,a[-1])*der_sigmoid(z[-1])
        for l in range(2,L):
            delta[-l] = np.dot(w[-l+1].T,delta[-l+1])*der_sigmoid(z[-l])
        grad_w = [np.dot(delta[l],a[l-1].T) for l in range(1,L)]
        grad_b = delta[1:]
        return (grad_w,grad_b)

    def gradCost(self,y,a):
        return (a-y)

    def evaluate(self,test):
        aux = [(np.argmax(self.evalOutput(x)), np.argmax(y)) for x,y in test]
        return sum(int(x==y) for (x,y) in aux)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def der_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def printNum(x,y):
    plt.imshow(x.reshape((28,28)),cmap='binary')
    plt.title(str(np.argmax(y)))
    plt.show()
