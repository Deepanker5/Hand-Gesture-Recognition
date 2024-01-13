import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dictionary = pickle.load(open('./training_data.pickle', 'rb'))
data = np.asanyarray(data_dictionary['data'])
letters = np.asanyarray(data_dictionary['letters'])

x_train, x_test, y_train, y_test = train_test_split(data, letters, test_size=0.2, shuffle=True, stratify=letters)
test_model = RandomForestClassifier()

test_model.fit(x_train, y_train)
y_predict = test_model.predict(x_test)
file = open('chosen_model.p', 'wb')
pickle.dump({'chosen_model': test_model}, file)
file.close
