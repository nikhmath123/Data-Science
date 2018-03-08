""" Installed SciKit dependency. 
	CHALLENGE: Use any 3 SciKit-Learn Models on Dataset, compare results, and print best one.
"""

#import classifiers
from sklearn import tree 
from sklearn import svm 
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

#DATA: [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'female', 'female', 'female', 'male', 
	 'male', 'male', 
	 'female', 'male', 'female', 'male']
	 
#Decision Tree Classifier	 
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X,Y) 
#Support Vector Classifier
clf_svm = svm.SVC()							 
clf_svm = clf_svm.fit(X,Y) 
#Gaussian process Classifier
clf_nb = GaussianProcessClassifier() 	 
clf_nb = clf_nb.fit(X,Y)

#Predictions
prediction_tree = clf_tree.predict([[190,70,43]])
print(prediction_tree)  
prediction_svm = clf_svm.predict([[190,70,43]]) 
print(prediction_svm)
prediction_nb = clf_nb.predict([[190,70,43]])
print(prediction_nb)

best_prediction = max(prediction_tree,prediction_svm,prediction_nb)
#Compares all 3 Models
if best_prediction == prediction_tree:
	print("Decision Tree gives most accurate prediction.")
elif best_prediction == prediction_svm:
	print("Support Vector Machine gives most accurate prediction.")
elif best_prediction == prediction_nb:
	print("Naive Bayes gives most accurate prediction.")


