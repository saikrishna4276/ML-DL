from tpot import TPOTClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

#load digits and iris dataset
digits_data = load_digits()
iris_data = load_iris()


X_digits = digits_data['data']
y_digits = digits_data['target']

X_iris = iris_data['data'] 
y_iris = iris_data['target']

#split the data into training and testing sets

X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, 
                                                                                stratify=y_digits, random_state=42)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris,
                                                                        stratify=y_iris, random_state=42)
#initialize tpot classifier
classifier = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

#fit the classifier on digits training data
classifier.fit(X_train_digits, y_train_digits)
#fit the classifier on iris training data
classifier.fit(X_train_iris, y_train_iris)


