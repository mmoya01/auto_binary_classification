from utils import *
from scale_features import *

from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold
from sklearn.utils import class_weight


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import f1_score,recall_score
from sklearn.utils import class_weight

import time


from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])
    
    def get_values(self, metric_name, window):
        d =  pd.Series(self.log_values[metric_name])
        return d.rolling(window,center=False).mean()

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))    




class auto_binary_classifier:
    def __init__(self,df,target_column,num_output_features,n_rows=None):
        self.df = df
        self.target_column = target_column
        self.num_output_features = num_output_features
        self.n_rows = n_rows

    def data_preprocessing(self,tree_based):
        """
        This function this returns the dataframe with one hot encoded/label encoded features, dimensionally reduced features, 
        dropping features that have weird distributions, scaling continous variables and 
        imputing any nan(and dropping columns that have too many nans)

        Parameters
        ----------
        df : pandas dataframe
            this is the data

        target_column : string
            the target variable name

        tree_based  :  boolean
            set to False by default. Set tree_based=True if you're working with a tree based model(i.e RF, xgboost,etc.)        
            
        num_output_features  :  int
            number of output features desired as far as dimensionality reduction
            
        dimension_reduction_method  :  string    
            'autoencoder' or 'feature_selection'
        n_rows  :  int
            number of rows desired as far as random sampling. Default is set to n_rows=None

        Returns
        -------
        dataframe
            this returns a tranformed dataframe ready for train/test split 
        """
       
        df = drop_useless_columns(self.df,get_categorical_cols(self.df,self.target_column))

        #if the string data is already converted to int, just OHE for non tree models or keep as is for tree models
        if len(list({k:v for (k,v) in df.dtypes.items() if v == 'object'}.keys())) and tree_based==False:
            df = ohe_data(df,get_categorical_cols(df,self.target_column))
        else:
            df = MultiColumnLabelEncoder(get_categorical_cols(df,self.target_column)).fit_transform(df)
        
      
        #drop any columns that have too many categories(i.e if there are 1000 rows and 800 categories for a column, drop that column
         #drop useless columns like Id where the number of unique categories = the number of rows in the entire dataframe   
        for col in list(df):
            if len(set(df[col]))/df.shape[0] > 0.8:
                df = df.drop([col],1)
            else:
                pass
              
        #if there are no categorical variables, we don't need to worry about encoding strings
#         if len(get_categorical_cols(df,self.target_column))!= 0:
#             if tree_based is False:
#                 df = ohe_data(df, get_categorical_cols(df,self.target_column))
#             else:
                
#                 df = MultiColumnLabelEncoder(get_categorical_cols(df,self.target_column)).fit_transform(self.df)
#         else:
#             pass

        #scale continuous features
#         columns_to_scale = list({k:v for (k,v) in df.dtypes.items() if v == 'float'}.keys())
        df[list(set(list(df))-set([self.target_column]))] = df[list(set(list(df))-set([self.target_column]))].apply(lambda x: (x-x.min())/(x.max()-x.min()))
        #impute missing values and drop any columns that are mostly nan   
        df = inpute_nan(df)
        #reduce dimensionality   
        df = feature_selection(df,self.target_column,self.num_output_features,tree_based)
        
        return df
    
    def binary_class_resample(self):
        """
        This function is used to rebalance the sample. By default,n_rows=None, it'll down sample the majority class based on the
        the number of rows in the minority class(note: sampling is done randomly). The user may also specify the number of rows per class

        Parameters
        ----------
        df : pandas dataframe
            this is the data

        target_column : string
            the target variable name

        n_rows  :  int
            this is used to specify the number of rows you would like to randomly per class

        Returns
        -------
        dataframe
            this returns a down sampled dataframe. If n_rows = None, this will always return a 50/50 balanced sample
        """
        if self.n_rows==None:
            class_imbalance = check_binary_imbalance(self.df,self.target_column)
            #randomly sample and reduce the majority class
            n_rows = self.df[self.df[self.target_column]!=class_imbalance['majority_class']].shape[0]
            majority = self.df[self.df[self.target_column]==class_imbalance['majority_class']].reset_index(drop=True)
            majority = majority.sample(n=n_rows, random_state=0)
            #now let's consider minority class and concat with majority dataframe
            minority_class = self.df[self.df[self.target_column]!=class_imbalance['majority_class']].reset_index(drop=True)
            self.df = pd.concat([majority,minority_class]).reset_index(drop=True)
        else:
            classes = list(set(self.df[self.target_column]))
            df1 =  self.df[self.df[self.target_column]==classes[0]].sample(n=self.n_rows, random_state=0)
            df2 =  self.df[self.df[self.target_column]==classes[1]].sample(n=self.n_rows, random_state=0)
            self.df = pd.concat([df1,df2]).reset_index(drop=True)
        return self.df
    
    def preprocess_imbalance(self,tree_based):
        """
        combines both data_preprocessing and binary_class_resample methods
        and alternative to this would be just doing data_preprocessing() and then SMOTE
        SMOTE functionality (if preferred to random random undersampling) will be incorporated at the modeling level
        """
        df = self.binary_class_resample()
        df = self.data_preprocessing(tree_based)
        return df
    
    
    def choose_binary_model(self):
        """
        This method is used to choose the best binary classification
        
        Returns a dictionary that highlights the best classifier and class imbalance(SMOTE or random undersample) technique
        Non tree based models will be OHE and every model will utilize feature importance as far as dimensionality reduction
        i.e
        
        {'sampling_method': 'random_undersample',
         'classifier': 'xgboost',
         'evaluation_metric': 'recall',
         'std': 0.02,
         'mean_accuracy': 0.56,
         'execution_time': 7.393228530883789}
        
        
        Note: this can be modified using the autoencoder found in scale_features.py instead(I ran out of time as far as figuring
        out why it was underperforming)
        """
        classifiers = [LogisticRegression(),SVC(),RandomForestClassifier(),xgb.XGBClassifier()]
        labels = ['logistic_regression','support_vector_machine','random_forest','xgboost']
        min_max_scaler = preprocessing.MinMaxScaler()
        cv = StratifiedKFold(n_splits=5)
        
        

        result = pd.DataFrame()
        # first check to see if there is imbalance
        imbalance_ratio = check_binary_imbalance(self.df,self.target_column)['imbalance_ratio']
        for model_type in ['sklearn','keras']:
            if model_type == 'sklearn':
                pass
                #--------------------------------------------------------------------------------
                #-------------------- sklearn class imbalance ----------------------------------
                #------------------------------------------------------------------------------

                if imbalance_ratio > 0:
                    accuracy = 'recall'
                    for sampling_method in ['random_undersample','SMOTE']: 

                        #--------------------- random undersample method --------------------------------
                        if sampling_method == 'random_undersample':

                            for clf, label in zip(classifiers, labels):
                                start_time = time.time()
                                if label in ['random_forest','xgboost']:
                                    data = self.preprocess_imbalance(True)
                                else:
                                    data = self.preprocess_imbalance(False)
                                
                                

                                X = np.array(data.drop([self.target_column],1))
                                y = np.array(data[self.target_column])
                                scores = cross_val_score(clf, X, y, cv=5, scoring=accuracy)
                                r = pd.DataFrame({'sampling_method':[sampling_method],
                                                  'classifier':[label],
                                                  'evaluation_metric':[accuracy],
                                                  'std':[np.round(scores.std(),2)],
                                                  'mean_accuracy':[np.round(scores.mean(),2)],
                                                  'execution_time':[time.time() - start_time]})
                                result = pd.concat([result,r]).reset_index(drop=True)
                        # ------------------------ SMOTE Method -------------------------------------
                        else:
                            accuracy = 'recall'
                            for clf, label in zip(classifiers,labels):
                                start_time = time.time()
                                if label in ['random_forest','xgboost']:
                                    data = self.data_preprocessing(True)
                                else:
                                    data = self.data_preprocessing(False)

                                X = np.array(data.drop([self.target_column],1))
                                y = np.array(data[self.target_column])

                                scores = []
                                for train_idx, test_idx, in cv.split(X, y):
                                    X_train, y_train = X[train_idx], y[train_idx]
                                    X_test, y_test = X[test_idx], y[test_idx]
                                    X_train, y_train = SMOTE().fit_sample(X_train, y_train)
                                    clf.fit(X_train, y_train)
                                    scores.append(recall_score(y_test,clf.predict(X_test)))
                                r = pd.DataFrame({'sampling_method':[sampling_method],
                                                  'classifier':[label],
                                                  'evaluation_metric':[accuracy],
                                                  'std':[np.round(np.std(scores),2)],
                                                  'mean_accuracy':[np.round(np.mean(scores),2)],
                                                  'execution_time':[time.time() - start_time]})
                                result = pd.concat([result,r]).reset_index(drop=True)
                #---------------------------------------------------------------------------------
                #-------------------- sklearn no class imbalance ----------------------------------
                #----------------------------------------------------------------------------------
                else:
                    accuracy = 'accuracy'
                    for clf, label in zip(classifiers, labels):
                        start_time = time.time()
                        if label in ['random_forest','xgboost']:
                            data = self.data_preprocessing(True)
                        else:
                            data = self.data_preprocessing(False)
    
                        X = np.array(data.drop([self.target_column],1))
                        y = np.array(data[self.target_column])
                        scores = cross_val_score(clf, X, y, cv=5, scoring=accuracy)
                        r = pd.DataFrame({'sampling_method':['None'],
                                                  'classifier':[label],
                                                  'evaluation_metric':[accuracy],
                                                  'std':[np.round(scores.std(),2)],
                                                  'mean_accuracy':[np.round(scores.mean(),2)],
                                          'execution_time':[time.time() - start_time]})
                        result = pd.concat([result,r]).reset_index(drop=True)

             #-----------------------------------------------------------------------
             #-------------------- Keras Model --------------------------------------
             #-----------------------------------------------------------------------               
            else:
                start_time = time.time()
                if imbalance_ratio > 0:
                    evaluation_metric = 'recall'
                    data = self.preprocess_imbalance(False)
                    
                else:
                    evaluation_metric = 'accuracy'
                    data = self.data_preprocessing(False)
                    
                X = data.drop([self.target_column],1)
                y = data[self.target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


                class_weights = {0: imbalance_ratio,
                                1: 1-imbalance_ratio
                                }

                input_dim = X_train.shape[1]
                model = Sequential()
                model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1,  activation='sigmoid'))

                if evaluation_metric == 'recall':
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[recall_m])
                else:
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m])

                history = model.fit(
                              np.array(X_train), np.array(y_train),
                              batch_size=128, epochs=32,verbose=1,callbacks=[BatchLogger()],
                              validation_data=(np.array(X_test), np.array(y_test)))
                score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
                false_positive_rate, recall, thresholds = roc_curve(y_test, model.predict(X_test)[:,0])
            
                r = pd.DataFrame({'sampling_method':['random_undersample'],
                                          'classifier':['keras_seq_model'],
                                          'evaluation_metric':[evaluation_metric],
                                          'std':[np.std(recall)],
                                          'mean_accuracy':[np.round(score[1],2)],
                                  'execution_time':[time.time() - start_time]})
                result = pd.concat([result,r]).reset_index(drop=True)
                
               

        result.to_pickle('binary_classification_results.pkl')
        final_result = result.sort_values('mean_accuracy',ascending=False).reset_index(drop=True).head(2)
        final_result = final_result[final_result['std']==np.min(final_result['std'])].iloc[0].to_dict()
        return final_result
    
    
