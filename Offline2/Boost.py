import numpy as np

def adaboost(X, Y, Learner, args):
    """
    X: training set features
    Y: training set labels
    Learner: learner class
    args: arguments
    """
    k = args.number_of_learners
    n = len(X)
    h = [] #list of learners
    z = [] #list of weights
    w = [1/n for i in range(n)] #weights of training examples
    # p = 0
    # for i in range(n):
    #     if Y[i] == 1:
    #         p += 1
    # w = [1/p if Y[i] == 1 else 1/(n-p) for i in range(n)]
    # w = [w[i]/sum(w) for i in range(n)]
    eps = 1e-6
    
    print("Training...")
    print(len(X), "examples")
    print(len(Y), "labels")
    print(k, "learners")
    print("\n--------------------\n")
    
    max_iter = 100
    iter = 0
    i = 0

    def select_top_samples(X, Y, w, limit, allPositive=False, allNegative=False):
        if limit <= 0:
            return [], [], []
        if allPositive:
            if allNegative:
                return X, Y, w
            else:
                retX, retY, retW, remX, remY, remW = [], [], [], [], [], []
                for i in range(len(X)):
                    if Y[i] == 1:
                        retX.append(X[i])
                        retY.append(Y[i])
                        retW.append(w[i])
                    else:
                        remX.append(X[i])
                        remY.append(Y[i])
                        remW.append(w[i])
                remX, remY, remW = select_top_samples(remX, remY, remW, limit - len(retX), False, False)

                return np.concatenate([retX, remX]), np.concatenate([retY, remY]), np.concatenate([retW, remW])
        
        if allNegative:
            retX, retY, retW, remX, remY, remW = [], [], [], [], [], []
            for i in range(len(X)):
                if Y[i] == 0:
                    retX.append(X[i])
                    retY.append(Y[i])
                    retW.append(w[i])
                else:
                    remX.append(X[i])
                    remY.append(Y[i])
                    remW.append(w[i])
            remX, remY, remW = select_top_samples(remX, remY, remW, limit - len(retX), False, False)

            return np.concatenate([retX, remX]), np.concatenate([retY, remY]), np.concatenate([retW, remW])
        limit = min(limit, len(X))
        temp = np.concatenate([X, np.array(Y).reshape(-1, 1), np.array(w).reshape(-1, 1)], axis=1)
        ord = np.argsort(temp[:, -1])
        ord = reversed(ord)
        temp = temp[list(ord)]
        return temp[:limit, :-2], temp[:limit, -2], temp[:limit, -1]
        # w = [w[i]/sum(w) for i in range(len(w))]
        # ord = np.random.choice(len(X), limit, p=w)
        # return X[ord], Y[ord], w[ord]

    while i < k:
        # print("Learner", i + 1)
        #resample
        Xnew, Ynew, w_new = select_top_samples(X, Y, w, args.dataset_size, args.all_positive, args.all_negative)
        h.append(Learner(Xnew, Ynew, w_new, args))
        # h.append(Learner(X, Y, w, args))
        error = 0
        res = []
        for j in range(n):
            res.append(h[i].predict(X[j]))
            if res[j] != Y[j]:
                error += w[j]
        # print("Learner", i + 1, "error:", error)
        if error > 0.5:
            iter += 1
            
            # print(error)
            h.pop()
            if iter == max_iter:
                print("Training failed")
                exit(1)
            continue
        iter = 0
        error = max(error, eps)
        for j in range(n):
            if res[j] == Y[j]:
                w[j] *= error/(1-error)
        
        w = [w[j]/sum(w) for j in range(n)]
        z.append(np.log((1-error)/error))
        i += 1
        # print("Learner", i, "error:", error)
    
    z = [z[i]/sum(z) for i in range(len(z))]
    return h, z