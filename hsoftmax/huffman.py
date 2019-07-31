# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:49:35 2019

@author: admin
"""

from random import shuffle
from copy import copy

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

# turns a list to a binary tree
def random_binary_full_tree(outputs):
    outputs = copy(outputs)
    shuffle(outputs)   # 随机排序

    while len(outputs) > 2:
        temp_outputs = []
        for i in range(0, len(outputs), 2):
            if len(outputs) - (i+1) > 0:
                temp_outputs.append([outputs[i], outputs[i+1]])
            else:
                temp_outputs.append(outputs[i])
        outputs = temp_outputs
    return outputs

################# hierarchical softmax class ########################################
import dynet as dy
import data

class hier_softmax:
    def __init__(self, tree, contex_size, model):
        self._tree_tools = TreeTools()
        self.str2weight = {}
        #create a weight matrix and bias vector for each node in the tree
        for i, subtree in enumerate(self._tree_tools._get_subtrees(tree)):
            self.str2weight["softmax_node_"+str(i)+"_w"] = model.add_parameters((len(subtree), contex_size))
            self.str2weight["softmax_node_" + str(i) + "_b"] = model.add_parameters(len(subtree))
        
        #create a dictionary from each value to its path
        value_to_path_and_nodes_dict = {}
        for path, value in self._tree_tools._get_leaves_paths(tree):
            nodes = self._tree_tools._get_nodes(tree, path)
            value_to_path_and_nodes_dict[data.char2int[value]] = path, nodes
        self.value_to_path_and_nodes_dict = value_to_path_and_nodes_dict
        self.model = model
        self.tree = tree
    
    #get the loss on a given value (for training)
    def get_loss(self, context, value):    #value为类别标签权重
        loss = []
        path, nodes = self.value_to_path_and_nodes_dict[value]
        for p, n in zip(path, nodes):
            w = dy.parameter(self.str2weight["softmax_node_"+str(n)+"_w"])
            b = dy.parameter(self.str2weight["softmax_node_" + str(n) + "_b"])
            probs = dy.softmax(w*context+b)
            loss.append(-dy.log(dy.pick(probs, p)))
        return dy.esum(loss)
    
############### Lets add hierarchical softmax to the model ######################################
output_tree = random_binary_full_tree(data.characters)

RNN_BUILDER = dy.LSTMBuilder
class AttentionNetworkWithHierSoftmax:
    
    def __init__(self, enc_layers, dec_layers, embeddings_size, enc_state_size, dec_state_size, tree):
        self.model = dy.Model()
        
        # the embedding paramaters
        self.embeddings = self.model.add_lookup_parameters((data.VOCAB_SIZE, embeddings_size))

        # the rnns
        self.ENC_RNN = RNN_BUILDER(enc_layers, embeddings_size, enc_state_size, self.model)
        self.DEC_RNN = RNN_BUILDER(dec_layers, enc_state_size, dec_state_size, self.model)
        
        # attention weights
        self.attention_w1 = self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = self.model.add_parameters((1, enc_state_size))
        
        self.enc_state_size = enc_state_size 
        
        self.hier_softmax = hier_softmax(tree, dec_state_size, self.model)
    
    def _get_probs(self, rnn_output, output_char):
        return self.hier_softmax.get_loss(rnn_output, output_char) 
        
    def _predict(self, rnn_output):
        return self.self.hier_softmax.generate(rnn_output)   
        
