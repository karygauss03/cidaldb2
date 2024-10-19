#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
class Split_data:
    def __init__ (self,test_size  ):
        self.test_size = test_size
    def split (self, x , y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=1)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
        return (x_train, y_train, x_test, y_test, x_val, y_val)

