import numpy as np

# a neural network
class net_work():
    def __init__(self,node,lr=0.2):
        self.lay = len(node)
        self.weight = list()
        self.lr = lr
        self.b = list()
        self.node = node
        self.loss = 0.0
        self.error = 0.0
        self.act_res = list()
        for i in range(self.lay-1):
            tmp = np.random.random((self.node[i],self.node[i+1]))-0.5
            self.weight.append(tmp)
            self.b.append(0)
        for i in self.node:
            self.act_res.append(np.zeros(i))

# the autoencoder architecture
class autoencoder():
    def __init__(self):
        self.encoder = list()
    def one_more(self,net_work):
        self.encoder.append(net_work)



