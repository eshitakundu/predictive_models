from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#classifiers:
clf_DTC = DecisionTreeClassifier() #machine learning model that classifies data
clf_RFC = RandomForestClassifier()
clf_SVC = SVC() #support vector classifier
clf_KNN = KNeighborsClassifier() #K-Nearest Neighbors Classifier

#dataset:

# [height, weight, shoe_size]
X = [[180, 60, 44], [172, 55, 40], [185, 70, 42], [168, 58, 39], [176, 62, 41], 
     [182, 65, 43], [170, 57, 38], [178, 63, 40], [188, 75, 45], [175, 61, 42], 
     [183, 68, 44], [169, 56, 39], [177, 64, 41], [186, 72, 43], [171, 59, 38], 
     [179, 66, 40], [187, 73, 45], [173, 60, 39], [181, 67, 42], [174, 62, 40], 
     [184, 69, 43], [167, 55, 38], [189, 74, 45], [172, 61, 39], [180, 68, 42], 
     [176, 63, 41], [182, 70, 43], [168, 57, 38], [178, 64, 40], [185, 71, 42], 
     [171, 58, 39], [179, 65, 41], [187, 74, 44], [173, 59, 38], [181, 66, 42], 
     [188, 75, 45], [174, 61, 40], [183, 69, 43], [169, 56, 38], [177, 63, 41], 
     [186, 72, 44], [170, 58, 39], [178, 64, 40], [184, 70, 43], [175, 62, 40], 
     [182, 67, 42], [189, 76, 45], [186, 72, 44], [185, 66, 42], [184, 70, 43]
]
# corresponding gender
Y = ['male', 'female', 'male', 'female', 'male', 
     'male', 'female', 'male', 'male', 'female', 
     'male', 'female', 'male', 'male', 'female', 
     'male', 'male', 'female', 'male', 'male', 
     'female', 'male', 'female', 'male', 'male', 
     'male', 'female', 'male', 'female', 'male', 
     'female', 'male', 'male', 'female', 'male', 
     'male', 'female', 'male', 'female', 'male', 
     'female', 'male', 'male', 'female', 'male', 
     'male', 'male', 'female', 'male', 'female'
]

#training the classifiers using given data (X, Y)
clf_DTC.fit(X, Y) 
clf_RFC.fit(X, Y)
clf_SVC.fit(X, Y)
clf_KNN.fit(X, Y)

#user input
try:
    height = int(input("Enter height (whole number, cm): "))
    weight = int(input("Enter weight (whole number, kg): "))
    shoe_size = int(input("Enter shoe size (whole number, EU): "))

    new = [[height, weight, shoe_size]]
    
except ValueError:
    print("Invalid input. Only numeric values allowed.")
    
prediction_DTC = clf_DTC.predict(new)
prediction_RFC = clf_RFC.predict(new)
prediction_SVC = clf_SVC.predict(new)
prediction_KNN = clf_KNN.predict(new)

print("Decision Tree Prediction: " , prediction_DTC) 
print("Random Forest Prediction: " , prediction_RFC) 
print("SVM Prediction:", prediction_SVC) 
print("KNN Prediction:", prediction_KNN) 
