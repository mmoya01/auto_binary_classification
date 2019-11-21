import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Input
from tensorflow.keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


'''--------------------------- Feature Importance -------------------------------------'''

def feature_selection(df,target_column,num_output_features,tree_based=False):
    """this function chooses the top n features, where n = num_output_features, that is most predictive.
    This method sets tree_based to False, switch to tree_based=True if you're using a tree based model
    I chose a logistic regression since it's an easy/computationally efficient classification algorithm
    for non tree based algorithms that use OHE
    
    Parameters
    ----------
    df : pandas dataframe
        pass in the dataframe of interest
        
    target_column : string
        this is the defined target variable name
        
    num_output_features  :  int
        this is the number of features you would like to reduce by
    
    tree_based  :  boolen
        True or False. True if you're using some sort of tree based model, False else
    
    Returns
    -------
    dataframe
        returns transformed dataframe based on the num_output_features mentioned above
    """
    X = df.drop([target_column],1)
    y = df[target_column]
    
    if tree_based == False:
        clf = LogisticRegression()
        clf = clf.fit(X, y)
        values = [abs(i) for i in clf.coef_[0]]
        top_features = [sorted(list(zip(list(X), values)), key=lambda x: x[1],reverse=True)[j][0] for j in range(len(values))][:num_output_features]
    else:
        clf = ExtraTreesClassifier()
        clf.fit(X, y)
        top_features = [sorted(list(zip(list(X), clf.feature_importances_)), key=lambda x: x[1],reverse=True)[j][0] for j in range(len(clf.feature_importances_))][:num_output_features]

    return df[top_features+[target_column]]



'''-------------------- dimensionality reduction ---------------------------------------------------'''
def autoencoder_dim_reduce(df,target_column,num_output_features):
    """this function reduces the dimensionality of your data by creating latent representations of 
     (binary or continous) features from the original dataset. 

     If m is the number of rows, n is the 
     number of original features and p = num_output_features, then the data is transformed from
     dimension (n,m) to (n,p)

    Parameters
    ----------
    df : pandas dataframe
        pass in the dataframe of interest

    target_column : string
        this is the defined target variable name

    num_output_features  :  int
        this is the number of features you would like to reduce by. For example, let's say you have 500 features but you 
        would like your final

    Returns
    -------
    dataframe
        returns transformed dataframe based on the num_output_features mentioned above
    """    
    # Choose size of our encoded representations (we will reduce our initial features to this number)
    encoding_dim = num_output_features

    #split training and test data
    train, test_df = train_test_split(df, test_size = 0.15, random_state= 1984)
    train_df, dev_df = train_test_split(train, test_size = 0.15, random_state= 1984)


    train_y = train_df[target_column]
    dev_y = dev_df[target_column]
    test_y = test_df[target_column]

    train_x = train_df.drop([target_column], axis = 1)
    dev_x = dev_df.drop([target_column], axis = 1)
    test_x = test_df.drop([target_column], axis = 1)

    train_x =np.array(train_x)
    dev_x =np.array(dev_x)
    test_x = np.array(test_x)

    train_y = np.array(train_y)
    dev_y = np.array(dev_y)
    test_y = np.array(test_y)

    #------------------------------------Build the AutoEncoder------------------------------------


    # Define input layer
    input_data = Input(shape=(train_x.shape[1],))
    # Define encoding layer
    encoded = Dense(encoding_dim, activation='elu')(input_data)
    # Define decoding layer
    decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)
    # Create the autoencoder model
    autoencoder = Model(input_data, decoded)
    #Compile the autoencoder model
    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy')
    #Fit to train set, validate with dev set and save to hist_auto for plotting purposes
    hist_auto = autoencoder.fit(train_x, train_x,
                    epochs=5,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(dev_x, dev_x))


    # Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
    encoder = Model(input_data, encoded)

    # Create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Encode and decode our test set (compare them vizually just to get a first insight of the autoencoder's performance)
    encoded_x = encoder.predict(test_x)
    decoded_output = decoder.predict(encoded_x)

    #--------------------------------Build new model using encoded data--------------------------
    #Encode data set from above using the encoder
    encoded_train_x = encoder.predict(train_x)
    encoded_test_x = encoder.predict(test_x)

    model = Sequential()
    model.add(Dense(16, input_dim=encoded_train_x.shape[1],
                    kernel_initializer='normal',
                    #kernel_regularizer=regularizers.l2(0.02),
                    activation="relu"
                    )
              )
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='adam')

    history = model.fit(encoded_train_x, train_y, validation_split=0.2, epochs=15, batch_size=64)
    
    encoded_train_df = pd.DataFrame(encoder.predict(train_x))
    encoded_train_df[target_column] = train_y

    encoded_test_df = pd.DataFrame(encoder.predict(test_x))
    encoded_test_df[target_column] = test_y

    encoded_dev_df = pd.DataFrame(encoder.predict(dev_x))
    encoded_dev_df[target_column] = dev_y

    transformed_data =pd.concat([encoded_train_df,encoded_test_df,encoded_dev_df])
    
    #---------------------------------Evaluate loss and ROC-----------------------
    # Plot history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Encoded model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
    predictions_NN_prob = model.predict(encoded_test_x)
    predictions_NN_prob = predictions_NN_prob[:,0]
    predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
    acc_NN = accuracy_score(test_y, predictions_NN_01)

    #Print Area Under Curve
    false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
    roc_auc = auc(false_positive_rate, recall)
    plt.figure()
    plt.title('Autoencoder ROC')
    plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out (1-Specificity)')
    plt.show()
    
    return transformed_data


