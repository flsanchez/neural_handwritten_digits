import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import cPickle as pickle

def genData(num_name,lab_name,idx_init,idx_fin):

    """
    genData(num_name,lab_name,idx_init,idx_fin,out_name) takes the "number" raw
    data from "num_name", the "label" raw data from "lab_name" and pickle the
    output data in the "out_name" file; with data within the indexes
    "idx_init" and "idx_fin".
    """

    """
    num_name="train-images-idx3-ubyte.gz"
    lab_name="train-labels-idx1-ubyte.gz"
    """
    n = idx_fin - idx_init

    with gzip.open(num_name,'rb') as f:
        f.seek(16+idx_init*28*28)
        data = f.read()
        data = [struct.unpack(">B",data[i])[0] for i in range(len(data))]
        #numbers = [data[28*28*i:28*28*(i+1)]/255.0 for i in range(n)]
        numbers = [np.array(data[28*28*i:28*28*(i+1)],dtype="float32")/255.0 for i in range(n)]
        #numbers = [np.reshape(np.array(data[28*28*i:28*28*(i+1)]),(28,28)) for i in range(n)]

    with gzip.open(lab_name,'rb') as f:
        f.seek(8+idx_init)
        label = f.read()
        labels = [struct.unpack(">B",label[i])[0] for i in range(n)]


    #data = zip(numbers,labels)
    data = [numbers,labels]
    #pickle.dump(data, open(out_name,'wb'))

    return data

    """
    #code for plot
    for i in range(n):
        plt.figure(i)
        plt.imshow(numbers[i], cmap='binary')
        plt.title("Numero {}".format(labels[i]))
    plt.show()
    #"""


out_name = "training_validation_test.pkl"
num_name=["train-images-idx3-ubyte.gz","t10k-images-idx3-ubyte.gz"]
lab_name=["train-labels-idx1-ubyte.gz","t10k-labels-idx1-ubyte.gz"]

data = []
print "Training Data"
data.append(genData(num_name[0],lab_name[0],0,50000))
print "Validation Data"
data.append(genData(num_name[0],lab_name[0],50000,60000))
print "Test Data"
data.append(genData(num_name[1],lab_name[1],0,10000))
print "Saving data..."
gz = gzip.open(out_name +'.gz','wb')
gz.write(pickle.dumps(data,pickle.HIGHEST_PROTOCOL))# ,open(out_name,'wb'),pickle.HIGHEST_PROTOCOL))
gz.close()
