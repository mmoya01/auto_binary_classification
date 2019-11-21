<h1> auto_binary_classifier</h1>
<h4>Please feel free to explore demo.ipynb in regards to examples of the components in this package</h4>
The auto_binary_classifier class is an all encompassing binary classification that allows a user to automatically:

* <b>Preprocess their data</b>
    * one hot or label encoding (particularly for tree vs non tree based models)
        * Unless if there is some type of ordinal relationship, I try to avoid label encoding for non tree sklearn models. For Keras, I normally work with word embedding layers or arrays of label encoded strings(or multilabel biniarizer)
    * it determines whether the data has class imbalance
    * drop columns with weird distributions such as _pymnt_plan_ and _initial_list_status_
    * drops columns that have to many categories(such as Id)
    * imputes missing values and drops columns with too many missing values
    
* <b>Adjusts for class imbalance issues</b>
    * if the program determines that there is significant class imbalance, it will automatically adjust this for user by considering:
        * random undersampling - it does a 50/50 undersample split by default, but the user can specify number of rows, n_rows
        * SMOTE
    * it then computes which class imbalance technique yields the best recall
    
* <b>Dimensionality reduction</b>
    * allows the user to choose the desired number of features. This is particularly important for cases in which one hot encoding is involved
    * Note: I built out an autoencoder in addition to a feature selection(I personally avoid PCA as far as binary features). I ended up choosing feature selection by default because the autoencoder kept underperforming in all cases
    * other methods for dimension reduction include TNSE and UMAP
    
* <h4>Lastly, it automatically chooses the best Model(and the best sampling technique)</h4>  
    * This package considers the following (vanilla)classifiers: 
        * Logistic Regression
        * SVM
        * Random Forest
        * Xgboost
        * Sequential model via Keras
    * for each model I evaluation k=5 folds and consider the model with the highest accuracy(or recall if class imbalance) and lowest deviation in scores    
    * The end result looks something like this
    
    ```{'sampling_method': 'random_undersample',
     'classifier': 'xgboost',
     'evaluation_metric': 'recall',
     'std': 0.03,
     'mean_accuracy': 0.61,
     'execution_time': 1.1010563373565674}
     ```    
    * Moreover, we can check out the results of all other models within binary_classification_results.pkl
        
    
<p><b>Things I would've love to build out if I had more time:</b></p> 
* after finding the best model, I would like to do grid search to optimize it more
* UMAP, particularly for sparse matrices. So instead of just comparing sampling methods by model, I would also test out more dimensionality techniques. I tend to prefer using latent representation of information rather than removal
* adjusting for multiclass problems. I started building out the functionality to determine class imbalance on a multiclass level and automate understanding the distribution for each class but I ran out of time

* unit tests - due to time constraints, I was mostly using other uci datasets such as their default credit dataset or the titanic dataset to weed out any bugs

