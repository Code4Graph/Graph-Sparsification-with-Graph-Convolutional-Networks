## Graph Sparsification with Graph Convolutional Networks

1. run __train.py__ inside the folder __pretrain_scgn__ (original code from kipf) to get the pretrain weights which are saved as npy files.

2. put the weights npy files into the folder __sgcn__ and run the __train-auto-admm-tuneParameter.py__ which does sparsification training and saves the sparsified adjacency matrix as __adj_0 matrix.npy__.

3. put the __adj_0 matrix.npy__ into the folder __gcn+sgcn__ and run the __train2.py__ which shows the performance of sgcn_gcn.
