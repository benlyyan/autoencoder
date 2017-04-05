import numpy as np
from auto_art import autoencoder,net_work

# activation function
def sigmoid(x):
    return 1./(1+np.exp(-x))

# forward propagation
def forward(net_work,x,y):
    lay = net_work.lay
    num = x.shape[0]
    net_work.act_res[0] = x
    for i in range(1,lay):
        # print net_work.act_res[i-1].shape,net_work.weight[i-1].shape
        net_work.act_res[i] = sigmoid(np.dot(net_work.act_res[i-1], net_work.weight[i-1])+net_work.b[i-1])
    net_work.error = y - net_work.act_res[lay-1]
    net_work.loss = 1./2.*(net_work.error**2).sum()/num
    return net_work

# backward propagation
def backward(net_work):
    lay = net_work.lay
    delta = list()
    for i in range(lay):
        delta.append(0)
    delta[lay-1] = -net_work.error*net_work.act_res[lay-1]*(1-net_work.act_res[lay-1])
    for i in range(1,lay-1)[::-1]:
        delta[i] = np.dot(delta[i+1],net_work.weight[i].T)*net_work.act_res[i]*(1-net_work.act_res[i])

    for i in range(lay-1):
        net_work.weight[i] -= net_work.lr*np.dot(net_work.act_res[i].T, delta[i+1])/(delta[i+1].shape[0])
        net_work.b[i] -= net_work.lr*delta[i+1]/(delta[i+1].shape[0])
    return net_work

# train a network
def train_network(net_work,x,y,iter):
    for i in range(iter):
        forward(net_work, x, y)
        backward(net_work)
        if i%50==0:
            print("=========iteration:%d=========" % i)
            res = net_work.act_res[-1]>=0.5
            accuracy = np.mean(res==y)*100
            print("Training accuracy:%f" % accuracy)
    return net_work

# build the autoencoder architecture
def form_antoencoder(node,lr=0.5):
    lay = len(node)
    en = autoencoder()
    for i in range(lay-1):
        en.one_more(net_work([node[i],node[i+1],node[i]],lr))
    return en

# train the encoder one by one
def train_encoder(en,x,iter):
    num = len(en.encoder)
    for i in range(num):
        en.encoder[i] = train_network(en.encoder[i], x, x, iter)
        tmp = forward(en.encoder[i], x, x)
        x = tmp.act_res[1]
    return en

