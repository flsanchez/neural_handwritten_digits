import numpy as np
import gzip
import cPickle

def loadData():
    print "Leyendo Data...",
    f = gzip.open("training_validation_test.pkl.gz",'rb')
    data,validation,test = cPickle.load(f)
    f.close()
    print "\t OK!"
    data = transformData(data)
    validation = transformData(validation)
    test = transformData(test)
    return [data,validation,test]

def transformData(data):
    x = [i.reshape((784,1)) for i in data[0]]
    y = []
    for i in data[1]:
        aux = np.zeros((10,1))
        aux[i] = 1
        y.append(aux)
    return [x,y]
