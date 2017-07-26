import loadData as ld
#import mnist_loader as mld
a,b,c = ld.loadData()
#a,b,c = mld.load_data_wrapper()
import net
red = net.network([784,30,10])
"""a = zip(*a)
b = zip(*b)
c = zip(*c)"""
red.SGD(a,3.0,10,30,b)
"""a = zip(a[0],a[1])
b = zip(b[0],b[1])
data = b
for i in range(3):
    print max(data[i][0])
    net.printNum(data[i][0],data[i][1])
#red.SGD2(a, 30, 10, 3.0, test_data=b)"""
