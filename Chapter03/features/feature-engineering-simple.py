from sklearn import preprocessing

'''Overall, this script demonstrates how to use scikit-learn's 
"OrdinalEncoder and OneHotEncoder to transform categorical data into 
"numerical formats suitable for machine learning models.
'''

#demonstrates how to use the preprocessing module from the sklearn (scikit-learn) library
# to encode categorical data

data = [['Bleach'], ['Cereal'], ['Toilet Roll']]

if __name__=="__main__":
    ordinal_enc = preprocessing.OrdinalEncoder()
    ordinal_enc.fit(data) #The fit method of this instance is called with the data list as an argument, which learns the ordinal encoding for the categorical data. 
    print(ordinal_enc.transform(data))

    onehot_enc = preprocessing.OneHotEncoder()
    onehot_enc.fit(data)
    print(onehot_enc.transform(data).toarray())


