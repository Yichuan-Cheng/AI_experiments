import numpy as np
from collections import Counter
from math import log
import copy
from graphviz import Digraph
import time
import uuid
from sklearn.metrics import classification_report,accuracy_score
train_data=np.loadtxt('traindata.txt')
test_data=np.loadtxt('testdata.txt')

#计算信息熵
def calculate_ent(dataSet):
    num=len(dataSet)
    labels={}
    for row in dataSet:
        label =row[-1]
        if label not in labels:
            labels[label]=0
        labels[label]+=1
    Ent=0.0
    for key in labels:
        prob =float(labels[key])/num
        Ent-=prob*log(prob,2)
    return Ent

def get_features_labels(dataset):
    return dataset[:,:-1],dataset[:,-1]


class node():#小于阈值去左子树，大于阈值去右子树
    def __init__(self) -> None:
        self.name=str(uuid.uuid1())
        self.lchild=None
        self.rchild=None
        self.feature=None
        self.thresh=0
        self.label=None
        self.dataset=None
        self.gain=0

    def split(self):
        features,labels=get_features_labels(self.dataset)
        if(labels.std()==0):
            self.label=labels[0]
            return
        features_num=len(features[0])
        dataset_ent=calculate_ent(self.dataset)
        feature=0
        thresh=0
        gain=0
        for i in range(features_num):
            feature_use=features[:,i]
            feature_unique_values=list(set(feature_use))
            split_points=[]
            feature_unique_values=sorted(feature_unique_values)
            for j in range(len(feature_unique_values)-1):
                split_points.append((feature_unique_values[j]+feature_unique_values[j+1])/2)
            for point in split_points:
                ldataset=[]
                rdataset=[]
                for row in self.dataset:
                    if(row[i]<point):
                        ldataset.append(row)
                    else:
                        rdataset.append(row)
                ldataset_ent=calculate_ent(ldataset)*(len(ldataset)/len(self.dataset))
                rdataset_ent=calculate_ent(rdataset)*(len(rdataset)/len(self.dataset))
                if(dataset_ent-(ldataset_ent+rdataset_ent)>gain):
                    gain=dataset_ent-(ldataset_ent+rdataset_ent)
                    feature=i
                    thresh=point
        if(gain>0):
            ldataset=[]
            rdataset=[]
            for row in self.dataset:
                if(row[feature]<thresh):
                    ldataset.append(row)
                else:
                    rdataset.append(row)
            self.thresh=thresh
            self.feature=feature
            self.gain=gain
            lnode=node()
            rnode=node()
            lnode.dataset=np.array(ldataset)
            rnode.dataset=np.array(rdataset)
            self.lchild=lnode
            self.rchild=rnode
            return
        else:
            label_list=list(labels)
            self.label=max(label_list, key=label_list.count)
            return
    def predict(self,features):
        if (self.gain==0):
            return self.label
        judge_feature=features[self.feature]
        if(judge_feature<self.thresh):
            return self.lchild.predict(features)
        else:
            return self.rchild.predict(features)
def loop(node_origin):
    node_origin.split()
    if(node_origin.gain==0):
        return 
    loop(node_origin.lchild)
    loop(node_origin.rchild)
def draw_tree(g,tnode):
    if(tnode.gain==0):
        g.node(name=tnode.name,label=str(tnode.label),color='green')
    else:
        g.node(name=tnode.name,label='feature:'+str(tnode.feature)+'\nthresh:'+str(tnode.thresh),color='black')
        g.node(name=tnode.lchild.name)
        g.node(name=tnode.rchild.name)
        g.edge(tnode.name,tnode.lchild.name,color='red')
        g.edge(tnode.name,tnode.rchild.name,color='blue')
        draw_tree(g,tnode.lchild)
        draw_tree(g,tnode.rchild)
def get_acc(dtree):
    prediction=[]
    label=[]
    for i in test_data:
        label.append(i[-1])
        prediction.append(dtree.predict(i[:-1]))
    return accuracy_score(label,prediction)

dt=node()
dt.dataset=np.array(train_data)
loop(dt)


g = Digraph('决策树')
draw_tree(g,dt)
g.view()

acc=get_acc(dt)
node_list=[]
def count_nodes(dtree):
    if(dtree.gain!=0):
        node_list.append(dtree)
        count_nodes(dtree.lchild)
        count_nodes(dtree.rchild)
count_nodes(dt)
iters=len(node_list)
for i in range(iters):
    test_tree=copy.deepcopy(dt)
    node_list=[]
    count_nodes(test_tree)
    features,labels=get_features_labels(node_list[i].dataset)
    node_list[i].gain=0
    label_list=list(labels)
    node_list[i].label=max(label_list, key=label_list.count)
    acc_=get_acc(test_tree)
    if(acc_>=acc):
        print('cut  '+"acc_ %f"%acc_)
        break
g = Digraph('剪枝决策树')
draw_tree(g,test_tree)
g.view()

prediction=[]
label=[]
for i in test_data:
    label.append(i[-1])
    prediction.append(test_tree.predict(i[:-1]))
print('-------------------------------------------------------------------------------')
print(classification_report(label,prediction))
print("label",Counter(label))
print("model_predictions",Counter(prediction))
