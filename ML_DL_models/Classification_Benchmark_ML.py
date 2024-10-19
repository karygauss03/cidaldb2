#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, f1_score
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

class Classification_Benchmark :
    def __init__ (self,model_name, fp_name, data_state):
        self.model_name = model_name
        self.fp_name = fp_name
        self.data_state = data_state
    def fit(self,x_train, y_train, export = False, name = "test_model"):
        model_fit = None
        if self.model_name == "RF" :
            model=RandomForestClassifier(n_estimators=150,max_features = "log2")
            model_fit =model.fit(x_train, y_train)
        elif self.model_name == "SVM":
            model = SVC(random_state=42,probability=True)
            model_fit =model.fit(x_train, y_train)
        elif self.model_name == "LR":
            model_fit = LogisticRegressionCV(cv=5, solver = 'lbfgs', max_iter = 650, multi_class = "ovr", n_jobs = -1).fit(x_train, y_train)
        elif self.model_name == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(100,50,25),
                            max_iter = 100,activation = 'relu',
                            solver = 'adam')
            model_fit =model.fit(x_train,y_train)
        elif self.model_name == "NB" :
            model= GaussianNB()
            model_fit =model.fit(x_train,y_train)
        elif self.model_name == "GB" :
            model= GradientBoostingClassifier(loss = 'exponential', criterion = 'squared_error')
            model_fit =model.fit(x_train,y_train)

        if export == True :
            filename = f'./{name}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(model_fit, file)
        train_scores = []
        pred_train = model_fit.predict(x_train)
        train_scores.append(accuracy_score(y_train, pred_train))
        train_scores.append(precision_score(y_train, pred_train))
        train_scores.append(recall_score (y_train, pred_train))
        train_scores.append(f1_score(y_train, pred_train))
        train_scores.append (matthews_corrcoef (y_train, pred_train))
        train_scores.append(balanced_accuracy_score (y_train, pred_train))
        train_scores.append(roc_auc_score(y_train,model_fit.predict_proba(x_train)[:, 1]))
        ConfusionMatrixDisplay.from_predictions(y_train, pred_train,cmap= "Blues")
        all_sample_title = 'Train dataset'
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_{}_confusion_matrix_train.png".format(self.data_state,self.model_name, self.fp_name)
        plt.savefig(filename, dpi=300)  
        return (model_fit, train_scores)
    
    def evaluate (self,model_fit,x_val, y_val, x_test, y_test, train_scores ) :
        valid_scores = []
        test_scores = []
        
        pred_val = model_fit.predict(x_val)
        valid_scores.append(accuracy_score(y_val, pred_val))
        valid_scores.append(precision_score(y_val, pred_val))
        valid_scores.append(recall_score (y_val, pred_val))
        valid_scores.append(f1_score(y_val, pred_val))
        valid_scores.append (matthews_corrcoef (y_val, pred_val))
        valid_scores.append(balanced_accuracy_score (y_val, pred_val))
        valid_scores.append(roc_auc_score(y_val, model_fit.predict_proba(x_val)[:, 1]))
        ConfusionMatrixDisplay.from_predictions(y_val, pred_val,cmap= "Blues")
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'validation dataset'
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_{}_confusion_matrix_validation.png".format(self.data_state, self.model_name, self.fp_name)
        plt.savefig(filename, dpi=300)   
       
        pred_test = model_fit.predict(x_test)
        test_scores.append(accuracy_score(y_test, pred_test))
        test_scores.append(precision_score(y_test, pred_test))
        test_scores.append(recall_score (y_test, pred_test))
        test_scores.append(f1_score(y_test, pred_test))
        test_scores.append (matthews_corrcoef (y_test, pred_test))
        test_scores.append(balanced_accuracy_score (y_test, pred_test))
        test_scores.append(roc_auc_score(y_test,model_fit.predict_proba(x_test)[:, 1]))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_test,cmap="Blues")
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'test dataset'
        plt.title(all_sample_title, size = 15)
        filename = "{}_{}_{}_confusion_matrix_test.png".format(self.data_state, self.model_name, self.fp_name)
        plt.savefig(filename, dpi=300)   
        metric_names = ["Accuracy","Precision","Recall","F1_score","MCC","Balanced_accuracy","ROC-AUC"]
        train_df = pd.DataFrame({"values": train_scores, "Metrics": metric_names})
        train_df["dataset"] = "train_scores"

        valid_df= pd.DataFrame({"values": valid_scores, "Metrics": metric_names})
        valid_df["dataset"] = "valid_scores"

        test_df = pd.DataFrame({"values": test_scores, "Metrics": metric_names})
        test_df["dataset"] = "Test_scores"

        df = pd.concat([train_df,valid_df, test_df])
        df = df.pivot(index="Metrics", columns="dataset", values="values")
        df = df.reset_index()
        df.insert(0, "Model Name", self.model_name)
        df.insert(0, "FP", self.fp_name)
        df.insert(0, "DATA ", self.data_state)

        df.to_csv('{}_{}_{}_metric_results.csv'.format(self.data_state, self.model_name, self.fp_name), index=False)

        return  (df ,pred_val, pred_test)


# In[ ]:




