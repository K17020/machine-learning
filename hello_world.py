from sklearn import tree

# Classifer for Apple and Oranges
# Training Data
# Features for Apples and Oranges

# Inputs, Weight(grams), Texture 1 = smooth & 0 = bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Outputs , Label 0 = apple & 1 = orange 
labels = [0, 0, 1, 1]

# Training Classifer, This is a box of rules
clf = tree.DecisionTreeClassifier()

# learning algorithm for tree Trains the classifer
clf = clf.fit(features, labels)

print (clf.predict([[160, 1]]))