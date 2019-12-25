import pandas as pd
from sklearn import model_selection
import pickle

#DATA PREPERATION
df = pd.read_csv("CreditCard_test.csv", skiprows=1)
#loading training data

(rows,cols) = df.shape

df = df.drop(['ID'], axis=1)

# Split-out validation dataset
array = df.values
X = array[:,0:23]
Y = array[:,23]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=7)

filename = 'finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
