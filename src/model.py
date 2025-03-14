from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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
    inp = np.load('./data/in_0.npy')
    out = np.load('./data/out_0.npy')
    # for i in range(1, 10):
    #     inp = np.concatenate( (inp, np.load(f'./data/in_{i}.npy')) )
    #     out = np.concatenate( (out, np.load(f'./data/out_{i}.npy')) )

    print(f'Dataset Loaded ({inp.shape[0]} examples)')
    return inp, out

X, y = loadDataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = tree.DecisionTreeClassifier(max_depth=200)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
preddist = np.zeros((9,))
acc = 0
successes = []
for X_testi, y_predi, y_testi in zip(X_test, y_pred, y_train):
    if y_predi == y_testi:
        successes += [(X_testi, y_predi)]
    preddist[y_predi] += 1

print("Distribution of outputs:", preddist/len(y_pred))
print("Accuracy on Test Set:", len(successes)/len(y_pred))

for success in successes[:10]:
    nodeind = clf.decision_path(success[0])
    print(nodeind) # u can remove this
    # maxim comes in here