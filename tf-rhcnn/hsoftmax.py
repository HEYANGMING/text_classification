# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 04:17:01 2019

@author: heyangm
"""

####################### define some auxiliary functions to handle trees #####################
class TreeTools:  
    def __init__(self):
        #memoization for _count_nodes functions
        self._count_nodes_dict = {}
                
    def _get_subtrees(self, tree):
        yield tree
        for subtree in tree:
            if type(subtree) == list:
                for x in self._get_subtrees(subtree):
                    yield x
                    
    # Returns pairs of paths and leafves of a tree
    def _get_leaves_paths(self, tree):
        for i, subtree in enumerate(tree):
            if type(subtree) == list:
                for path, value in self._get_leaves_paths(subtree):
                    yield [i] + path, value
            else:
                yield [i], subtree
    
    # Returns the number of nodes in a tree (not including root nodes)
    def _count_nodes(self, tree):
        if id(tree) in self._count_nodes_dict:
            return self._count_nodes_dict[id(tree)]
        size = 0
        for node in tree:
            if type(node) == list:
                size += 1 + self._count_nodes(node)
        self._count_nodes_dict[id(self._count_nodes_dict)] = size
        return size

    # Returns all the nodes in a path
    def _get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        for decision in path:
            nodes.append(next_node)
            next_node += 1 + self._count_nodes(tree[:decision])
            tree = tree[decision]
        return nodes


################# hierarchical softmax class ########################################
import dynet as dy
import data

class h_softmax:
    def __init__(self, tree, contex_size, model):
        self._tree_tools = TreeTools()
        self.str2weight = {}
        
        #create a weight matrix vector for each node in the tree
        for i, subtree in enumerate(self._tree_tools._get_subtrees(tree)):
            self.str2weight["softmax_node_"+str(i)+"_w"] = model.add_parameters((len(subtree), contex_size))
        
        #create a dictionary from each value to its path
        value_to_path_and_nodes_dict = {}
        for path, value in self._tree_tools._get_leaves_paths(tree):
            nodes = self._tree_tools._get_nodes(tree, path)
            value_to_path_and_nodes_dict[data.char2int[value]] = path, nodes
        self.value_to_path_and_nodes_dict = value_to_path_and_nodes_dict
        self.model = model
        self.tree = tree
    
    #get the loss on a given value (for training)
    def get_predict(self, context, value):      #value为类别标签，即叶子节点
       
        path, nodes = self.value_to_path_and_nodes_dict[value]
        for n in zip(path, nodes):
            w = dy.parameter(self.str2weight["softmax_node_"+str(n)+"_w"])  #节点参数w
            return dy.softmax(w*context)
           