#HackON Hackathon
#==>> Topic = Health And Technology
#=====> "Health_Check":- Hacked by Team Avengers <======
#Members:-  1.Aniket Singh(#admin)
#           2.Pushkar Khadase(#ML Developer)
#           3.Somesh Lad(#Full Stack Developer)
def deeplearning():
    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import user_display
    
    #importing the data set
    dataset = pd.read_csv('dataset.csv')
    #splitting dataset into the independent(X) variable and dependent(Y) Variable
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,132].values
    
    #since there were missing data so we need to fill those data with the 0
    from sklearn.impute import SimpleImputer
    missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'constant', verbose = 0, fill_value=0)
    missingvalues = missingvalues.fit(X[:])
    X[:]=missingvalues.transform(X[:])
    
    #encoding the data
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    encoder = LabelEncoder()
    Y[:] = encoder.fit_transform(Y[:])
    Y = Y.reshape(-1,1)
    
    onehotencoder = OneHotEncoder(categorical_features = 'all')
    Y = onehotencoder.fit_transform(Y).toarray()

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 4926, random_state = 0)
    
    #building the artificial Neural network
    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    
    #Initialise the Classifier
    classifier = Sequential()
    
    #Adding the first input layer into the ANN
    classifier.add(Dense(output_dim = 87 , init = 'uniform' , activation = 'relu' , input_dim = 132 ))
    
    
    #Adding the final layer into the ANN as output Layer
    classifier.add(Dense(output_dim = 46 , init = 'uniform' , activation = 'softmax'))
    
    #compiling the model
    classifier.compile(optimizer = 'adam' ,loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    
    #fitting and training the model
    classifier.fit(X_train , y_train , batch_size = 10 , nb_epoch = 100)
    
    #predicting the results
    Y_pred = classifier.predict(X_test) 
    
    # #decoding the results
    pre = user_display.user_display()
    prediction = classifier.predict(np.array([pre]))
    
    prediction = onehotencoder.inverse_transform(prediction)
    
    prediction = encoder.inverse_transform(prediction.astype(int))
    return prediction
disease = deeplearning()
print("==>> your disease is  " + disease)
