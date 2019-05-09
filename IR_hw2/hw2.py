import numpy as np
import pickle as pk
import random
import sys

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(w, x):
    return sigmoid(np.dot(w,x))

def loss(w, x, z):
    return np.log(1+np.exp(np.dot(w,z)-np.dot(w,x)))

def gradient(w, x, z):
    return sigmoid(np.dot(w,z)-np.dot(w,x))*(z-x)

def valid(w, qid, feature, rel):
    with open("IDCG.pk", "rb") as fp:
        IDCG = pk.load(fp)
    nDCG = {}
    for q in qid:
        l = len(feature[q])
        wx = np.zeros(l)
        for i in range(1, l+1):
            wx[i-1] = f(w, feature[q][i])
        swx = np.argsort(wx)
        swx = swx[::-1]
        swx+=1
        r = np.zeros(l)
        for i in range(l):
            r[i] = rel[q][swx[i]]
        s = 0
        if len(r)<10:
            for i in range(1, len(r)+1):
                s += (np.power(2, r[i-1])-1) / np.log2(i+1)
        else:
            for i in range(1, 11):
                s += (np.power(2, r[i-1])-1) / np.log2(i+1)
        if IDCG[q]==0:
            nDCG[q] = 0
        else:
            nDCG[q] = s/IDCG[q]
    return sum(list(nDCG.values()))/len(nDCG)

def task1():
    with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
        train_rel = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
        train_feature = pk.load(fp)
    
    train_qid = list(train_rel.keys())[:6693]
    valid_qid = list(train_rel.keys())[6693:]
    print("===========Read data done===========")

    if len(sys.argv)>2:
        with open("weight.pk", "rb") as fp:
            w = pk.load(fp)
    else:
        w = np.random.rand(136)
        w = 2*w-1.0

    lr = 5e-3
    loss_list = []
    ndcg_list = []
    epoch = 400000
    bndcg = 0
    for e in range(1,epoch+1):
        lr *= 0.999995
        g = np.zeros(136)
        cnt = 0
        qid = random.choice(train_qid)
        maxr = np.max(list(train_rel[qid].values()))
        minr = np.min(list(train_rel[qid].values()))
        for i in train_rel[qid]:
            if train_rel[qid][i]==maxr:
                for j in train_rel[qid]:
                    if train_rel[qid][j]<maxr:
                        g += gradient(w,train_feature[qid][i],train_feature[qid][j])
                        cnt+=1
        if (maxr-minr)>1:
            for i in train_rel[qid]:
                if train_rel[qid][i]==(maxr-1):
                    for j in train_rel[qid]:
                        if train_rel[qid][j]<(maxr-1):
                            g += gradient(w, train_feature[qid][i],train_feature[qid][j])
                            cnt+=1
        if cnt>0:
            g = g/cnt
            w = w - lr*g
        #do validation
        if e%2000==0:
            ndcg = valid(w, valid_qid, train_feature, train_rel)
            #print("epoch:", e, " nDCG=", ndcg)
            cnt = 0
            cost = 0
            qqid = train_qid.copy()
            for qid in qqid:
                maxr = np.max(list(train_rel[qid].values()))
                minr = np.min(list(train_rel[qid].values()))
                for i in train_rel[qid]:
                    if train_rel[qid][i]==maxr:
                        for j in train_rel[qid]:
                            if train_rel[qid][j]<maxr:
                                cost += loss(w,train_feature[qid][i],train_feature[qid][j])
                                cnt+=1
            print("cost:", cost/cnt)
            #print("cost:", cost)
            loss_list.append(cost/cnt)
            ndcg_list.append(ndcg)

            if ndcg>bndcg:
                bndcg = ndcg
                with open("weight.pk", "wb") as fp:
                    pk.dump(w, fp, protocol=pk.HIGHEST_PROTOCOL)
            print("epoch:", e, "nDCG=", ndcg, "best nDCG=", bndcg)
    
    with open("loss_list.pk", "wb") as fp:
        pk.dump(loss_list, fp, protocol=pk.HIGHEST_PROTOCOL)
    with open("ndcg_list.pk", "wb") as fp:
        pk.dump(ndcg_list, fp, protocol=pk.HIGHEST_PROTOCOL)

def task2():
    with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
        train_rel = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
        train_feature = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/list_matrix.pk", "rb") as fp:
        list_matrix = pk.load(fp)

    train_qid = list(train_rel.keys())[:6693]
    valid_qid = list(train_rel.keys())[6693:]
    print("===========Read data done===========")

    if len(sys.argv)>2:
        with open("weight2.pk", "rb") as fp:
            w = pk.load(fp)
    else:
        w = np.random.rand(136)
        w = 2*w-1.0
    
    lr = 1e-4
    loss_list = []
    ndcg_list = []
    epoch = 10000
    bndcg = 0
    for e in range(1,epoch+1):
        lr = lr*0.999
        rqid = train_qid.copy()
        random.shuffle(rqid)
        for qid in rqid:
            ATA = list_matrix[qid][0]
            ATr = list_matrix[qid][1]
            g = np.dot(ATA, w) - ATr
            #w = w - lr*g/np.sqrt(len(train_rel[qid]))
            w = w - lr*g

        ndcg = valid(w, valid_qid, train_feature, train_rel)
        #print("epoch:", e, " nDCG=", ndcg)
        
        cost = 0
        for q in train_qid:
            for d in train_rel[q]:
                cost += (np.dot(w, train_feature[q][d])-train_rel[q][d])**2
        print("cost:", cost)
        loss_list.append(cost)
        ndcg_list.append(ndcg)

        if ndcg>bndcg:
            bndcg = ndcg
            with open("weight2.pk", "wb") as fp:
                pk.dump(w, fp, protocol=pk.HIGHEST_PROTOCOL)
        print("epoch:", e, "nDCG=", ndcg, "best nDCG=", bndcg)
        
    with open("loss_list2.pk", "wb") as fp:
        pk.dump(loss_list, fp, protocol=pk.HIGHEST_PROTOCOL)
    with open("ndcg_list2.pk", "wb") as fp:
        pk.dump(ndcg_list, fp, protocol=pk.HIGHEST_PROTOCOL)
    

def task3():
    with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
        train_rel = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
        train_feature = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/list_rel_matrix.pk", "rb") as fp:
        list_rel_matrix = pk.load(fp)
    train_qid = list(train_rel.keys())[:6693]
    valid_qid = list(train_rel.keys())[6693:]
    print("===========Read data done===========")

    if len(sys.argv)>2:
        with open("weight3.pk", "rb") as fp:
            w = pk.load(fp)
    else:
        w = np.random.rand(136)
        w = 2*w-1.0
    
    lr = 1e-4
    loss_list = []
    ndcg_list = []
    epoch = 2000
    bndcg = 0
    for e in range(1,epoch+1):
        rqid = train_qid.copy()
        random.shuffle(rqid)
        for qid in rqid:
            rATA = list_rel_matrix[qid][0]
            rATr = list_rel_matrix[qid][1]
            g = np.dot(rATA, w) - rATr
            w = w - lr*g

        ndcg = valid(w, valid_qid, train_feature, train_rel)
        #print("epoch:", e, " nDCG=", ndcg)

        cost = 0
        for q in train_qid:
            for d in train_rel[q]:
                cost += (train_rel[q][d]+1)*(np.dot(w,train_feature[q][d])-train_rel[q][d])**2
        print("cost:", cost)
        loss_list.append(cost)
        ndcg_list.append(ndcg)
        
        if ndcg>bndcg:
            bndcg = ndcg
            with open("weight3.pk", "wb") as fp:
                pk.dump(w, fp, protocol=pk.HIGHEST_PROTOCOL)

        print("epoch:", e, "nDCG=", ndcg, "best nDCG=", bndcg)

    with open("loss_list3.pk", "wb") as fp:
        pk.dump(loss_list, fp, protocol=pk.HIGHEST_PROTOCOL)
    with open("ndcg_list3.pk", "wb") as fp:
        pk.dump(ndcg_list, fp, protocol=pk.HIGHEST_PROTOCOL)

def listnet():
    with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
        train_rel = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
        train_feature = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/list_matrix.pk", "rb") as fp:
        list_matrix = pk.load(fp)
    train_qid = list(train_rel.keys())[:6693]
    valid_qid = list(train_rel.keys())[6693:]
    with open("/tmp2/b02902030/IR_hw2/X_query.pk", "rb") as fp:
        X_query = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/y_query_exp.pk", "rb") as fp:
        y_query_exp = pk.load(fp)
    print("===========Read data done===========")

    if len(sys.argv)>2:
        with open(sys.argv[2], "rb") as fp:
            w = pk.load(fp)
    else:
        w = np.random.rand(136)
        w = 2*w-1.0
   
    lamb = 0.9
    lr = 5e-3
    epoch = 10000
    G = np.zeros(136)
    ep = np.array([1e-4]*136)
    bndcg=0
    for e in range(1,epoch+1):
        #lr*=0.9999
        rqid = train_qid.copy()
        random.shuffle(rqid)
        for q in rqid:
            X = X_query[q]
            y = y_query_exp[q]
            H = np.exp(np.dot(X, w))
            XT = np.transpose(X)
            g = -np.dot(XT, y) + np.dot(XT, H) / np.sum(H)

            #add task2
            ATA = list_matrix[q][0]
            ATr = list_matrix[q][1]
            g = lamb*g + (1-lamb) * (np.dot(ATA, w) - ATr) / len(train_rel[q])
            
            #adagrad
            G += (g+ep)**2 
            g /= np.sqrt(G)

            w = w - lr*g


        ndcg = valid(w, valid_qid, train_feature, train_rel)

        if ndcg>bndcg:
            bndcg = ndcg
            with open("weight_list_moreloss_adagrad.pk", "wb") as fp:
                pk.dump(w, fp, protocol=pk.HIGHEST_PROTOCOL)

        print("epoch:", e, "nDCG=", ndcg, "best nDCG=", bndcg)


def task3_listnet():
    with open("/tmp2/b02902030/IR_hw2/train_rel.pk", "rb") as fp:
        train_rel = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/train_feature.pk", "rb") as fp:
        train_feature = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/list_matrix.pk", "rb") as fp:
        list_matrix = pk.load(fp)
    train_qid = list(train_rel.keys())[:6693]
    valid_qid = list(train_rel.keys())[6693:]
    with open("/tmp2/b02902030/IR_hw2/X_query.pk", "rb") as fp:
        X_query = pk.load(fp)
    with open("/tmp2/b02902030/IR_hw2/y_query_sigmoid.pk", "rb") as fp:
        y_query_sigmoid = pk.load(fp)
    print("===========Read data done===========")

    if len(sys.argv)>2:
        with open(sys.argv[2], "rb") as fp:
            w = pk.load(fp)
    else:
        w = np.random.rand(136)
        w = 2*w-1.0
   
    lamb = 0.9
    lr = 5e-2
    epoch = 10000
    G = np.zeros(136)
    ep = np.array([1e-4]*136)
    bndcg=0
    for e in range(1,epoch+1):
        #lr*=0.9999
        rqid = train_qid.copy()
        random.shuffle(rqid)
        for q in rqid:
            X = X_query[q]
            y = y_query_sigmoid[q]
            XT = np.transpose(X)
            wx = np.dot(X, w)
            S = sigmoid(wx)
            SP = S*(1-S)
            g = -np.dot(XT, y*(1-S)) + np.dot(XT, SP) / np.sum(S)

            #add task2
            #ATA = list_matrix[q][0]
            #ATr = list_matrix[q][1]
            #g = lamb*g + (1-lamb) * (np.dot(ATA, w) - ATr) / len(train_rel[q])
            
            #adagrad
            G += (g+ep)**2 
            g /= np.sqrt(G)

            w = w - lr*g


        ndcg = valid(w, valid_qid, train_feature, train_rel)

        if ndcg>bndcg:
            bndcg = ndcg
            with open("weight3_list_adagrad.pk", "wb") as fp:
                pk.dump(w, fp, protocol=pk.HIGHEST_PROTOCOL)

        print("epoch:", e, "nDCG=", ndcg, "best nDCG=", bndcg)





def task_test(weight_file, output_file):
    with open("/tmp2/b02902030/IR_hw2/test_feature.pk", "rb") as fp:
        feature = pk.load(fp)
    with open(weight_file, "rb") as fp:
        w = pk.load(fp)
    with open(output_file, "w") as fp:
        fp.write("QueryId,DocumentId\n")
        qid = list(feature.keys())
        for q in qid:
            l = len(feature[q])
            wx = np.zeros(l)
            for i in range(l):
                wx[i] = f(w, feature[q][i])
            swx = np.argsort(wx)
            swx = swx[::-1]
            swx = swx[:10]
            for i in swx:
                fp.write(str(q)+","+str(i)+"\n")




def main():
    if sys.argv[1] == "task1":
        task1()
    elif sys.argv[1] == "task2":
        task2()
    elif sys.argv[1] == "task3":
        task3()
    elif sys.argv[1] == "listnet":
        listnet()
    elif sys.argv[1] == "task3_listnet":
        task3_listnet()
    elif sys.argv[1] == "test":
        task_test(sys.argv[2], sys.argv[3])

if __name__ == "__main__":
    main()
