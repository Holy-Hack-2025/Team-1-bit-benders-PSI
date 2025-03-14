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

feature_names = [
    "Metal Type: TIN", "Metal Type: STEEL", "Metal Type: IRON", "Item Type: RAW", "Item Type: SHEET", "Item Type: BLOCK",
    "Item Type: ROD", "Item Type: SCREW", "Item Type: SPRING", "Item Type: WASHER", "Item Type: CHASSIS",
    "Purity", "Hardness", "Coefficient of Thermal Expansion",
    "Priority", "Deviation", "Quality"
]

def extract_decision_path(tree, feature_names, sample):
    node_indicator = tree.decision_path(sample.reshape(1, -1))
    feature_idx = tree.tree_.feature
    threshold = tree.tree_.threshold
    decision_path = []

    for node_id in node_indicator.indices:
        if feature_idx[node_id] == -2 or feature_idx[node_id] >= len(feature_names):
            continue

        feature_name = feature_names[feature_idx[node_id]]
        threshold_value = threshold[node_id]
        sample_value = sample[feature_idx[node_id]]

        if sample_value <= threshold_value:
            decision_path.append(f"{feature_name} (value: {sample_value:.2f}) is ≤ {threshold_value:.2f}")
        else:
            decision_path.append(f"{feature_name} (value: {sample_value:.2f}) is > {threshold_value:.2f}")

    return decision_path



def generate_gpt_prompt_logistics(decision_path, predicted_class):
    """Formats a GPT-friendly prompt explaining the decision in bullet points."""
    prompt = "The decision tree made the following decision:\n"
    prompt += "- " + "\n- ".join(decision_path) + "\n"
    prompt += f"\nBased on the above conditions, the item was routed to line: {predicted_class}.\n"
    prompt += "\nPlease explain this decision in a professional yet easy-to-understand manner using bullet points for logistics given that they are interested in for example order completion efficiency ."

    return prompt

def generate_gpt_prompt_management(decision_path, predicted_class):
    """Formats a GPT-friendly prompt explaining the decision in bullet points."""
    prompt = "The decision tree made the following decision:\n"
    prompt += "- " + "\n- ".join(decision_path) + "\n"
    prompt += f"\nBased on the above conditions, the item was routed to line: {predicted_class}.\n"
    prompt += "\nPlease explain this decision in a professional yet easy-to-understand manner using bullet points for management given that they are interested in for example business productivity and cost minimization ."

    return prompt

def generate_gpt_prompt_engineering(decision_path, predicted_class):
    """Formats a GPT-friendly prompt explaining the decision in bullet points."""
    prompt = "The decision tree made the following decision:\n"
    prompt += "- " + "\n- ".join(decision_path) + "\n"
    prompt += f"\nBased on the above conditions, the item was routed to line: {predicted_class}.\n"
    prompt += "\nPlease explain this decision in a professional yet easy-to-understand manner using bullet points for engineering given that they are interested in for example machine integrity and product quality ."

    return prompt

for success in successes[:10]:
    #nodeind = clf.decision_path(success[0])
    #print(nodeind) # u can remove this
    sample_features, predicted_class = success
    decision_path = extract_decision_path(clf, feature_names, sample_features)
    gpt_prompt = generate_gpt_prompt_logistics(decision_path, predicted_class)
    print("\nGPT Prompt:\n")
    print(gpt_prompt)
    gpt_prompt = generate_gpt_prompt_management(decision_path, predicted_class)
    print("\nGPT Prompt:\n")
    print(gpt_prompt)
    gpt_prompt = generate_gpt_prompt_engineering(decision_path, predicted_class)
    print("\nGPT Prompt:\n")
    print(gpt_prompt)

#x_sample = randomX(1)
#print("Generated Feature Vector Shape:", x_sample.shape)
#print("Expected Feature Count from Dataset:", X.shape[1])
#print("Max feature index used by tree:", max(clf.tree_.feature))


