from sklearn import tree
from sklearn.model_selection import train_test_split
from structures import *

def randomX(amt):
    if amt == 1:
        item = Item(MetalType.IRON, ItemType.CHASSIS, .9, .9, 10., .9)
        req = Requirements((
            Order(Item(MetalType.IRON, ItemType.CHASSIS, .9, .9, 10., .9), 2),
            Order(Item(MetalType.STEEL, ItemType.CHASSIS, .9, .9, 10., .9), 3),
            Order(Item(MetalType.IRON, ItemType.CHASSIS, .9, .9, 10., .9), 0),
            Order(Item(MetalType.IRON, ItemType.CHASSIS, .9, .9, 10., .9), 0),
            Order(Item(MetalType.IRON, ItemType.CHASSIS, .9, .9, 10., .9), 0)
            ))

        return InputState(item, req).data()
    return np.array([randomX(1) for _ in range(amt)])

def randomY(amt):
    if amt == 1:
        return Decision(2).data()
    return np.array([randomY(1) for _ in range(amt)])

def genDataset(amt):
    return (randomX(amt), randomY(amt))

X, y = genDataset(400)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = tree.DecisionTreeClassifier(max_depth=10)

clf = clf.fit(X, y)

print(X[0], y[0])

print(clf.predict(X[0].reshape((1, -1))))