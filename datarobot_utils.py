import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
import re
from sklearn.base import TransformerMixin


'''
-----------------------get list of categorical variables ---------------------------------------
'''
def get_categorical_cols(df,target_column):
    """
    This function finds all of the categorical features 
    It also makes sure features have the correct data type
    i.e it treated emp_length as a string instead of int
    lastly,this function is able to understand that int64 columns like, mths_since_last_major_derog, are categorical

    Parameters
    ----------
    df : pandas dataframe
        this is the data
        
    target_variable : string
        this is the defined target variable name
        
    df  :  dataframe
        pass in the dataframe of interest

    Returns
    -------
    list
        this returns the list of categorical variables/features in the dataframe
    """
    omit_cols = [target_column]
    int_cols = []
    obj_cols = []

    schema_df = df.dtypes.to_dict()
    
    
    for key in list(schema_df.keys()):
        if str(schema_df[key]) == 'int64':
            int_cols.append(key)
        elif str(schema_df[key]) == 'object':
            obj_cols.append(key)
        else:
            pass
        
    # if the a col is int64 but it's more of category that has already been encoded(i.e if we had a column called 'rank')
    #the list,cat_int_cols, should catch such categorical variables in the event they need to be 
    # significantly smaller than the # rows, it is most likely a categorical variable
    # in this case, if the unique list of numbers is 10% or less than the number of rows, we assume it's a categorical variable 
    # in this case, "mths_since_last_major_derog" ends up being a categorical int64 column
    int_cols = [x for x in int_cols if x not in omit_cols]

    cat_int_cols = []
    for col in int_cols:
        if len(set(df[col]))/df.shape[0] < 0.1:
            cat_int_cols.append(col)
        else:
            pass

    categorical_cols = obj_cols+cat_int_cols
    
    return categorical_cols

'''---------------------------drop weird columns -----------------------------------------------------------'''
def drop_useless_columns(df,category_columns):
    """
    This function ends up dropping weird columns like pymnt_plan and initial_list_status. Moreover, it prints a message for other features
    that may need more attention
    
    Parameters
    ----------
    df : pandas dataframe
        this is the data
        
    category_columns : list
        this is the list of categorical variables in a dataframe which can be generated using get_categorical_cols
        
    Returns
    -------
    dataframe
        this returns the dataframe after dropping these weird columns
    """
    for cat in category_columns:
        cat_dist = sorted(list(Counter(df[cat]).values()))
        if np.std(cat_dist)/cat_dist[-1]>0.35:
            if len(cat_dist)<3:
                print('dropping column {}'.format(cat))
                df = df.drop([cat],1)
            else:
                category_issue = [list(Counter(df[cat]).values()).index(np.min(list(Counter(df[cat]).values())))]
#                 category_issue = list(Counter(df[cat]).keys())[list(Counter(df[cat]).values()).index(1)] 
                print('double check category {} from the column called {}'.format(category_issue,cat))
        else:
            pass
    return df

'''------------------- impute missing values(imputer located in datarobot_utils.py) -------------------------------------------'''
def inpute_nan(df):
    nan_cols  = df.isna().sum().to_dict()
    for key in nan_cols:
        #drop columns that have more than 50% missing values
        if nan_cols[key]/df.shape[0]>=0.5:
            df = df.drop([key],1)
        else:
            pass
    #the DataFrameImputer class can be found via datarobot_utils 
    df = DataFrameImputer().fit_transform(df)
    return df


'''---------------------------------------- one hot encode data -----------------------------------------------------------'''

def ohe_data(df,categorical_cols):
    """
    This function ends up one hot encodes all categorical variables.Since label encoding treats 0<1<2... 
    I will use one encoding for my sci-kit learn models
    
    Parameters
    ----------
    df : pandas dataframe
        this is the data
        
    category_columns : list
        this is the list of categorical variables in a dataframe which can be generated using get_categorical_cols
        
    Returns
    -------
    dataframe
        this returns the dataframe with one hot encoded features
    """
    for ohe in categorical_cols:
        df = pd.concat([df.drop(ohe, axis=1), pd.get_dummies(df[ohe],prefix='{}_'.format(ohe))], axis=1)
    return df

'''------------------------- labelencoder data(for tree based models) -------------------------------------------------------
note: I found this handy dandy class here: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
'''
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

'''
---------------------- check binary class imbalance ------------------------------------------------------------------------------
'''
def check_binary_imbalance(df,target_column):
    """this function checks to see if there is an issue with class imbalance for binary classification

    Parameters
    ----------
    df : pandas dataframe
        pass in the dataframe of interest
        
    target_column : string
        this is the defined target variable name

    Returns
    -------
    dict
        returns {'imbalance_ratio': float, 'majority_class': int}
        where imbalance_ratio is the ratio imbalance(if there is no imbalance then imbalance_ratio = 0.0) 
        and 'majority_class' is the majority class(if there is no significant imbalance, majority_class = NaN)
    """    
    class_counts = Counter(df[target_column])
    if (class_counts[0]/class_counts[1]<0.4)|(class_counts[1]/class_counts[0]<0.4):
        imbalance_ratio = np.round(np.min([(class_counts[0]/class_counts[1]),(class_counts[1]/class_counts[0])]),2)
        majority_class = list(class_counts.keys())[list(class_counts.values()).index(np.max(list(class_counts.values())))]
    else:
        imbalance_ratio = 0.0
        majority_class = np.nan
    
    class_imbalance = {'imbalance_ratio':imbalance_ratio,'majority_class':majority_class}
    return class_imbalance

'''-------------------------------------- check multiclass imbalance -------------------------------------------------------------'''
def check_multiclass_imbalance(df,target_column,min_multiclass_samplesize=None):
    """this function checks to see if there is an issue with class imbalance for multiclass classification

    Parameters
    ----------
    df : pandas dataframe
        pass in the dataframe of interest
        
    target_column : string
        this is the defined target variable name

    Returns
    -------
    dict
        returns {'imbalance_ratio': list, 'majority_class': list}
        where imbalance_ratio is the list of ratio imbalances(if there is no imbalance then imbalance_ratio = [0.0]) 
        and 'minority_class' is the minority class(if there is no significant imbalance, minority_class = [NaN])
    """ 
    
    if min_multiclass_samplesize is None:
        min_multiclass_samplesize = 50
    else:
        pass
    
    counts_by_cat = Counter(df[target_column])

    cat_dist = list(counts_by_cat.values())


    if np.min(cat_dist)/np.max(cat_dist)<0.4:   
        '''
        I will leave the min_class_sample cut off as an exploratory threshold for the user to decide given the distribution of 
        all classes. For different fields, such as medicine/clinical trials, practioners expect far smaller classes.
        The default will be set to a minimum of 50 samples
        '''
        check_min_samp = list(dict((k, v) for k, v in counts_by_cat.items() if v <= min_multiclass_samplesize).keys())
        #this condition looks at whether there are classes below 50 samples
        #if so, it will drop those rows and check to see if class imbalance still exists among other classes
        if len(check_min_samp)>0:
            #drop categories/rows that don't meet the min_class_sample cutoff
            df = df[~df[target_column].isin(check_min_samp)].reset_index(drop=True)
            counts_by_cat = Counter(df[target_column])
            cat_dist = list(counts_by_cat.values())
            max_val = np.max(cat_dist)
            without_max = [x for x in cat_dist if x not in [max_val]]
            ratios = [np.round(i/max_val,2) for i in without_max]

            #the condition below checks to see if there are any ratios less than 0.4
            if len([without_max[i] for i in list(np.where(np.array(ratios)<0.4)[0])])>0:
                imbalance_keys = [list(counts_by_cat.keys())[list(counts_by_cat.values()).index(j)] for j in [without_max[i] for i in list(np.where(np.array(ratios)<0.4)[0])]]
                imbalance_ratios = [ratios[i] for i in list(np.where(np.array(ratios)<0.4)[0])]
                class_imbalance = {'imbalance_ratios':imbalance_ratios,'minority_classes':imbalance_keys}
            else:
                class_imbalance = {'imbalance_ratios':[0.0],'minority_classes':[np.nan]}
                pass
        else:    
            max_val = np.max(cat_dist)
            without_max = [x for x in cat_dist if x not in [max_val]]
            ratios = [np.round(i/max_val,2) for i in without_max]
            imbalance_keys = [list(counts_by_cat.keys())[list(counts_by_cat.values()).index(j)] for j in [without_max[i] for i in list(np.where(np.array(ratios)<0.4)[0])]]
            imbalance_ratios = [ratios[i] for i in list(np.where(np.array(ratios)<0.4)[0])]
            class_imbalance = {'imbalance_ratios':imbalance_ratios,'minority_classes':imbalance_keys}
    else:
        class_imbalance = {'imbalance_ratios':[0.0],'minority_classes':[np.nan]}
    return class_imbalance


'''
----------------------- Exploratory Analysis for user: plots the distribution by category relative to the target variable -----------
'''

def plot_categorical_column(category_col,target_column,df):
    """I built this for data viz funsies. For instance, if someone wanted to visualize

    Parameters
    ----------
    category_col : string
        this is the category column that you would like to see the distribution for
        
    target_column : string
        this is the defined target variable name
        
    df  :  dataframe
        pass in the dataframe of interest

    Returns
    -------
    plot
        this returns a stacked bar chart of the results via matplotlib

    """
    total_dict = Counter(df[category_col])
    default_dict = Counter(df[df[target_column]==1][category_col])
    #the following sorts each dict by keys
    total = {key: total_dict[key] for key in sorted(total_dict.keys())}
    default = {key: default_dict[key] for key in sorted(default_dict.keys())}

    #it might be the case that the subset, where is_bad ==1, may not have a particular category
    # i.e home_owners where is_bad ==1 has no 'NONE' category
    #that said, we'll need to append count = 0 while preserving the index
    # list(total.keys()).index(key) will let us know which index to append on
    '''
    example:
    In[]:
    a = [1, 2, 4]
    insert_at = 2 
    a = a[:]   
    a[insert_at:insert_at] = [3]
    a

    Out[]:
    [1,2,3,4]

    '''
    total_counts =  list(total.values())
    default_counts = list(default.values())
    if set(list(total.keys())) != set(list(default.keys())):
        unmatched_keys = list(set(list(total.keys())) - set(list(default.keys())))
        for key in unmatched_keys:
            insert_at = list(total.keys()).index(key)
            default_counts = default_counts[:]
            #insert count == 0 at that index for that particular key
            default_counts[insert_at:insert_at] = [0]
    else:
        pass

    
    
    cat_labels = list(total.keys())
    le = LabelEncoder()
    le.fit(list(set(df[category_col])))
    cat_values = list(le.transform(le.classes_))
    plt.figure(figsize=(100,70))
    plt.bar(cat_labels,total_counts, color='m',alpha=0.5, label='Total Population')
    plt.bar(cat_labels,default_counts, color='b',alpha=0.5, label='Default')

    plt.xticks(cat_labels,rotation=90)
    plt.tick_params(labelsize=40)
    plt.ylabel('Number of People')
    plt.title('{} distribution'.format(' '.join(category_col.split('_'))),fontweight="bold",size=100)
    plt.legend(prop={'size': 80})
    plt.autoscale(enable=True) 
    return plt.show()

    
'''
------------------------------------------- Impute Missing Values--------------------------------------------------
I'm using a simple imputer found here : https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
all it does is it imputes a missing value with the most frequently occuring value
----------------------------------------------------------------------------------------------------------------------
'''    

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

