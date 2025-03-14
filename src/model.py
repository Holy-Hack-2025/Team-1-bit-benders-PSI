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
    return randomX(amt), randomY(amt)

def loadDataset():
    return np.load('./data/in.npy'), np.load('./data/out.npy')

X, y = loadDataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = tree.DecisionTreeClassifier(max_depth=10)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = 0
for y_predi, y_testi in zip(y_pred, y_test):
    acc += 1 if y_predi == y_testi else 0
print(acc/len(y_pred))
