#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import dgllife
# import torch.nn
import deepchem as dc
from deepchem.models.torch_models import MPNNModel
from deepchem.models.torch_models import AttentiveFPModel
from deepchem.models.torch_models import GCNModel
from deepchem.models.torch_models import GATModel
from deepchem.models import GraphConvModel
from deepchem.models import GraphConvTensorGraph 
# from deepchem.models import DAGModel
from deepchem.models.graph_models import DAGModel
from deepchem.molnet.preset_hyper_parameters import hps
from deepchem.metrics.metric import Metric
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
 

class Classification :  
    def __init__ (self , model_name, data_state) :
        self.model_name = model_name
        self.data_state = data_state

    def fit (self, train_dataset, valid_dataset,test_dataset,hyper_parameters=None, seed=123, nb_epoch = 40):
        if hyper_parameters is None and self.model_name not in ['gat', 'gcn','afp','mpnn']:
            hyper_parameters = hps[self.model_name] #import_hps_from_deepchem
        model = None
        if self.model_name == 'afp':
            model = AttentiveFPModel(n_tasks= 1, mode="classification", batch_size=32, learning_rate=0.001, droupout=0.1,model_dir ="{}_afp_model".format(self.data_state))    
        elif self.model_name == "gcn":
            model = GCNModel(n_tasks= 1, mode="classification", batch_size=32, learning_rate=0.001, dropout=0.1,model_dir="{}_gcn_model".format(self.data_state))
        elif self.model_name == 'gat' :
            model = dc.models.GATModel(n_tasks=1,mode="classification",batch_size=32, learning_rate=0.0001, dropout=0.25,model_dir= "{}_gat_model".format(self.data_state))
        elif self.model_name == "mpnn":
            #n_atoms_feat
            #n_pair_feat
            #n_hidden
            #n_graph_feat
            model = MPNNModel( n_tasks =1, mode="classification", batch_size=16, learning_rate=0.001,dropout = 0.25,model_dir="{}_mpnn_model".format(self.data_state))
        elif self.model_name == 'graphconv':
            batch_size = hyper_parameters['batch_size']
            nb_epoch = hyper_parameters['nb_epoch']
            learning_rate = hyper_parameters['learning_rate']
            n_filters = hyper_parameters['n_filters']
            n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

            model= GraphConvTensorGraph( 1,graph_conv_layers=[n_filters] * 2, dense_layer_size=n_fully_connected_nodes,
            batch_size=batch_size, learning_rate=learning_rate, random_seed=seed, mode='classification')
        elif self.model_name == 'dag':
            batch_size = hyper_parameters['batch_size']
            nb_epoch = hyper_parameters['nb_epoch']
            learning_rate = hyper_parameters['learning_rate']
            n_graph_feat = hyper_parameters['n_graph_feat']

            transformer = dc.trans.DAGTransformer(max_atoms=50)
            train_dataset = transformer.transform(train_dataset) 
            valid_dataset = transformer.transform(valid_dataset)
            test_dataset = transformer.transform(test_dataset)

            model = DAGModel(1,max_atoms=50,n_atom_feat=75, n_graph_feat=n_graph_feat,n_outputs=30,
            batch_size=batch_size, learning_rate=learning_rate, random_seed=seed, use_queue=False, mode='classification')
        model.fit(train_dataset, nb_epoch=nb_epoch)
        train_scores = {}
        valid_scores = {}
        test_scores = {}
        metric= [dc.metrics.roc_auc_score, dc.metrics.accuracy_score, dc.metrics.f1_score, dc.metrics.recall_score]
        train_scores[self.model_name] = model.evaluate(train_dataset, metric)
        valid_scores[self.model_name] = model.evaluate(valid_dataset, metric)
        test_scores[self.model_name] = model.evaluate(test_dataset, metric)
        return (model, train_scores, valid_scores , test_scores)

    def predict (self, model, train_dataset, train_dataset_y, valid_dataset,valid_dataset_y, test_dataset, test_dataset_y, train_scores, valid_scores , test_scores) :
        precision_scores = [self.model_name, "Precision"]
        mcc_scores = [self.model_name,"MCC"]
        balanced_acc_scores = [self.model_name,"Balanced_accuracy"]
#       __Train__
        predict_label_train = model.predict(train_dataset)
        label_train = [0] * len(predict_label_train)
        predict_label_train = list(predict_label_train)
        label_train = np.zeros(len(predict_label_train))
        for i in range(0, len(predict_label_train)):
            indx = np.where(predict_label_train[i] == np.amax(predict_label_train[i]))
            label_train[i] = indx[0]
        precision_train = precision_score(train_dataset_y, label_train, average='binary')
        precision_scores.append(precision_train)
        mcc_train = matthews_corrcoef (train_dataset_y, label_train)
        mcc_scores.append(mcc_train)
        balanced_acc_train = balanced_accuracy_score (train_dataset_y, label_train)
        balanced_acc_scores.append(balanced_acc_train)
        cr= classification_report(train_dataset_y, label_train)
        ConfusionMatrixDisplay.from_predictions(train_dataset_y, label_train, cmap = "Blues")
        all_sample_title = "train_dataset"
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_confusion_matrix_train.png".format(self.model_name,self.data_state)
        plt.savefig(filename, dpi=300)
#       __Validation__ 
        predict_label_val = model.predict(valid_dataset)
        label_val = [0] * len(predict_label_val)
        predict_label_val = list(predict_label_val)
        label_val = np.zeros(len(predict_label_val))
        for i in range(0, len(predict_label_val)):
            indx = np.where(predict_label_val[i] == np.amax(predict_label_val[i]))
            label_val[i] = indx[0]
        precision_val = precision_score(valid_dataset_y, label_val, average='binary')
        precision_scores.append(precision_val)
        mcc_val = matthews_corrcoef (valid_dataset_y, label_val)
        mcc_scores.append(mcc_val)
        balanced_acc_val = balanced_accuracy_score (valid_dataset_y, label_val)
        balanced_acc_scores.append(balanced_acc_val)
        cr= classification_report(valid_dataset_y, label_val)
        ConfusionMatrixDisplay.from_predictions(valid_dataset_y, label_val, cmap = "Blues")
        all_sample_title = "validation_dataset"
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_confusion_matrix_validation.png".format(self.model_name,self.data_state)
        plt.savefig(filename, dpi=300)
#        __Test__
        predict_label_test = model.predict(test_dataset)
        label_test = [0] * len(predict_label_test)
        predict_label_test = list(predict_label_test)
        label_test = np.zeros(len(predict_label_test))
        for i in range(0, len(predict_label_test)):
            indx = np.where(predict_label_test[i] == np.amax(predict_label_test[i]))
            label_test[i] = indx[0]
        precision_test = precision_score(test_dataset_y, label_test, average='binary')
        precision_scores.append(precision_test)
        mcc_test = matthews_corrcoef (test_dataset_y, label_test)
        mcc_scores.append(mcc_test)
        balanced_acc_test = balanced_accuracy_score (test_dataset_y, label_test)
        balanced_acc_scores.append(balanced_acc_test)
        cr= classification_report(test_dataset_y, label_test)
        ConfusionMatrixDisplay.from_predictions(test_dataset_y, label_test, cmap = "Blues")
        all_sample_title = "test_dataset"
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_confusion_matrix_test.png".format(self.model_name,self.data_state)
        plt.savefig(filename, dpi=300)
       
        # ___save___
        model_names = []
        metric_names = []
        train_scores_list = []
        valid_scores_list = []
        test_scores_list = []
        for model_name, metrics in train_scores.items():
            for metric_name, metric_value in metrics.items():
                model_names.append(model_name)
                metric_names.append(metric_name.replace("metric-1", "ROC-AUC").replace("metric-2", "Accuracy").replace("metric-3", "F1_score").replace("metric-4", "Recall"))
                train_scores_list.append(metric_value)
                valid_scores_list.append(valid_scores[self.model_name][metric_name])
                test_scores_list.append(test_scores[self.model_name][metric_name])
        metric_results = pd.DataFrame({'Model Name': model_names, 'metrics': metric_names, 'train_scores': train_scores_list, 'valid_scores': valid_scores_list, 'test_scores': test_scores_list})
        metric_results.loc[len(metric_results)]= precision_scores
        metric_results.loc[len(metric_results)]= mcc_scores
        metric_results.loc[len(metric_results)]= balanced_acc_scores
        metric_results.insert(0, "DATA", self.data_state)
        metric_results.to_csv('{}_{}_metric_results.csv'.format(self.model_name,self.data_state), index=False)
        metric_results
        return ( metric_results)

