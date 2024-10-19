#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deepchem as dc

class Split_data :
    def __init__ (self, split_type ) :
        self.split_type = split_type
    def splitter (self, dataset):
        if self.split_type == "random" :
            splitter = dc.splits.RandomSplitter()
        elif self.split_type == "random_str" :
            splitter = dc.splits.RandomStratifiedSplitter()
        elif self.split_type == "scaffold" :
            splitter = dc.splits.ScaffoldSplitter()
        train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed=123)
        return ( train_dataset, val_dataset, test_dataset)

