#Introduction - Learn Python for Data Science #1
#youtube link = https://www.youtube.com/watch?v=T5pRlIbr6gg&t=3s


from sklearn import tree
import graphviz


#[height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
 	 [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
 	 [171, 75, 42], [181, 85, 43]]



Y = ['male', 'female', 'female', 'female',
	 'male', 'male', 'male', 'female','male',
	 'female','male' ]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("gender_classifier")

prediction = clf.predict([[190,70,43]])
prediction_proba = clf.predict_proba([[190,70,43]])


# print clf
print prediction

print prediction_proba