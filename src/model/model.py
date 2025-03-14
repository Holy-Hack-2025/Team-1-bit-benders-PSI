from sklearn import tree
from sklearn.model_selection import train_test_split
from data.structure import *

def randomX(amt):
    if amt == 1:
        return Item(MetalType.IRON, ItemType.CHASSIS, .9, .8, 12., 1).data()
    return np.array([randomX(1) for _ in range(amt)])

def randomY(amt):
    pass

def genDataset(amt):
    return (randomX(amt), randomY(amt))

X, y = genDataset(400)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)
