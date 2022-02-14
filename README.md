## Graph Sparsification with Graph Convolutional Networks

This is the original TensorFlow implementation of SGCN in the following paper: 

[SGCN: A graph sparsifier based on graph convolutional networks, PAKDD 2020]

[Graph sparsification with graph convolutional networks, JDSA 2021]

### Pretrained step

1. run __train.py__ inside the folder __pretrain_scgn__ (original code from kipf) to get the pretrain weights which are saved as npy files.

### ADMM training step

2. put the weights npy files into the folder __sgcn__ and run the __train-auto-admm-tuneParameter.py__ which does sparsification training and saves the sparsified adjacency matrix as __adj_0 matrix.npy__.

### Testing

3. put the __adj_0 matrix.npy__ into the folder __gcn+sgcn__ and run the __train2.py__ which shows the performance of sgcn_gcn.
