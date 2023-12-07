def TestModel(X, Y, h, z, print_result=False):
    """
    X: test set features
    Y: test set labels
    h: list of learners
    z: list of weights
    """
    n = len(X)
    if print_result:
        print("Testing...")
        print(len(X), "examples")
        print(len(Y), "labels")
        print(len(h), "learners")
        print("\n--------------------\n")

    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(n):
        prediction = 0
        for j in range(len(h)):
            prediction += z[j]*(h[j].predict(X[i]) > 0.5)
        prediction = 1 if prediction > 0.5 else 0
        if prediction == Y[i]:
            if prediction == 1:
                TP += 1
            else:
                TN += 1
        else:
            if prediction == 1:
                FP += 1
            else:
                FN += 1


    accuracy = (TP+TN)/(TP+TN+FP+FN) 
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    false_discovery_rate = FP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    specificity = TN/(TN+FP) if TN+FP != 0 else 0

    f1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
    mcc = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5 if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0 else 0

    if print_result:
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"Specificity: {specificity:.6f}")
        print(f"False discovery rate: {false_discovery_rate:.6f}")
        print(f"F1: {f1:.6f}")
        print(f"MCC: {mcc:.6f}")
        print("\n--------------------\n")

    return accuracy, precision, recall, specificity, f1, mcc