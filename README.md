## Graph Sparsification with Graph Convolutional Networks
--------------------------------------------------------------------------------------------
This is the original TensorFlow implementation of SGCN in the following paper: 

[SGCN: A graph sparsifier based on graph convolutional networks, PAKDD 2020](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_22)

[Graph sparsification with graph convolutional networks, JDSA 2021](https://link.springer.com/content/pdf/10.1007/s41060-021-00288-8.pdf)

## Pretrained step
Run __train.py__ inside the folder __pretrain_scgn__ (original code from kipf) to get the pretrain weights which are saved as npy files.


## ADMM training step
Put the weights npy files into the folder __sgcn__ and run the __train-auto-admm-tuneParameter.py__ which does sparsification training and saves the sparsified adjacency matrix as __adj_0 matrix.npy__.


## Testing
Put the __adj_0 matrix.npy__ into the folder __gcn+sgcn__ and run the __train2.py__ which shows the performance of sgcn_gcn.
