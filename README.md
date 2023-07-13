# MPGNN
Here the repository for the paper "Meta-Path Graph Neural Networks for Heterogeneous Graphs".
![procedure.pdf](https://github.com/francescoferrini/MPGNN/files/11786872/procedure.pdf)

## Installation
The required packages to run the code are in [requirements](requirements.txt)

## Create synthetic graphs
In order to create synthetic graphs to train and test the model, navigate to ../data/final_datasets folder and run 
```sh
  python create_graph_multi_metapath_deterministic.py --num_nodes 5000 --max_rel_for_node 3 --metapath red-red-blue
```

* num_nodes: total number of nodes in the graph
* num_rel_types: total number of relation types in the graph
* max_rel_for_node: max out-degree for each node
* metapath: describe how the correct metapath is created. 

## Model
In order to run the model:
*  change the folder name of dataset (1046 of main.py)
*  specify which dataset to use in line 1037 and 1039 of main.py

Run with:
```sh
  mpiexec -n 5 python main.py
```
where:
* 5 is the total number of processes used
