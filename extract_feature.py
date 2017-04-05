def run_feature(input_data,label,nodescomplete,iters=10000):
    import auto_imp 
    from auto_art import autoencoder,net_work 
    import numpy as np 

    input_data = input_data
    label = label 
    nodescomplete = nodescomplete 
    iters = iters 

    print("="*40)
    print("Building autoencoder")

    nodes = nodescomplete[:-1]
    ae = auto_imp.form_antoencoder(nodes)
    ae = auto_imp.train_encoder(ae, input_data, 100)

    print("="*40)
    print("Training neural network.....")

    aecomplete = net_work(nodescomplete)

    # build a full network, initializing weights with trained autoencoder weights
    for i in range(len(nodescomplete)-2):
        aecomplete.weight[i] = ae.encoder[i].weight[0]

    aecomplete = auto_imp.train_network(aecomplete, input_data, label, iters)

    return aecomplete 

if __name__=="__main__":
    import numpy as np 
    input_data = np.loadtxt("/home/ben/Documents/Pyspace/auto_encoder/input_data.txt",delimiter=',',dtype=float)
    label = np.loadtxt("/home/ben/Documents/Pyspace/auto_encoder/label_of_inputdata.txt",dtype=float)
    label = label.reshape((len(label),1))
    iters = 1000
    nodescomplete = [input_data.shape[1],3,2,1]

    aecomplete = run_feature(input_data, label, nodescomplete,iters)

    print("="*40)
    print("Showing input data")
    print(aecomplete.act_res[0])

    print("Showing label data")
    print(label)

    print("="*40)
    print("Showing the first hidden layer results")
    print(aecomplete.act_res[1])

    print("="*40)
    print("Showing the second hidden layer results")
    print(aecomplete.act_res[2])

    print("="*40)
    print("Showing output of trained by neural network")
    print(aecomplete.act_res[-1])

    res = aecomplete.act_res[-1] >=0.5

    accuracy = np.mean(res==label)*100
    print("Traing accuracy:%f" % accuracy)
