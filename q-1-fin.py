#!/usr/bin/env python
# coding: utf-8

# In[530]:


import numpy as np
from IPython.display import Image,display
import matplotlib.pyplot as plt


# In[531]:


import numpy as np  
# import dill
import matplotlib.pyplot as plt  
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import math
import sys
from scipy.linalg import svd

# # Read dataset to pandas dataframe
# dataset = pd.read_csv('/home/aishwarya/CSIS/SMAI/SMAI_assig/a-5/codes/data.csv')
# dataset=dataset.iloc[:,1:]



loc='/home/aishwarya/CSIS/SMAI/SMAI_assig/asig_2/RobotDataset/Robot1'
dataset = pd.read_csv(loc,sep=' ') 
dataset.columns= ['Nan','target', 'c2', 'c3', 'c4','c5','c6','c7','id']

dataset = dataset.drop('id', 1)
dataset=dataset.dropna(axis=1)

#moving target to end
dataset=dataset[['c2', 'c3', 'c4','c5','c6','c7','target']]


train, validate = np.split(dataset, [int(.8*len(dataset))]) #for sequential data
# train, validate = np.split(dataset.sample(frac=1), [int(.8*len(dataset))]) # for random 

# train


# In[532]:


def prepare_data_X_Y(data1):
    X= data1.iloc[:,:-1].values
    Y= data1.iloc[:,-1].values
    return X,Y
    
x_train,Y_train=prepare_data_X_Y(train)
x_validate,y_validate=prepare_data_X_Y(validate)

# X=dataset1
# X=(X - dataset.mean().values)/dataset.std().values
# type(X)
# X
# x_train
#Xor data
XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
# X=XORdata[:,0:2]
# y=XORdata[:,-1]

X=x_train
y=Y_train
# len(X[0])
# x_validate[3]
ys=[3,5,3,2,3]
print len(np.unique(ys))


# In[533]:


def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


# In[534]:


def initialize_network(input_neurons,hidden_neurons,output_neurons,n_hidden_layers):
    weight_netwrk=[]       
    for i in range(n_hidden_layers):
        if i!=0:
            l=len(weight_netwrk[-1])
            input_neurons=l
            
        hidden_layer = [ { 'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons) ]
        weight_netwrk.append(hidden_layer)
    
    output_layer = [ { 'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    weight_netwrk.append(output_layer)
    
    return weight_netwrk


# In[537]:


input_neurons=len(X[0])
hidden_neurons=input_neurons+1
output_neurons=len(np.unique(y))

n_hidden_layers=int(raw_input("enter number of hidden layers"))

net=initialize_network(input_neurons,hidden_neurons,output_neurons,n_hidden_layers)

# print_network(net)


# In[538]:


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def activate_tanh(x):
    return np.tanh(x)

def activate_ReLU(x):
    return x * (x > 0)

def d_tanh(x):
    return 1. - x * x

def d_ReLU(x):
    return 1. * (x > 0)


# In[539]:


def forward_propagation(wgt_net,inputs,act_func):
    cur_row=inputs
    for layer in wgt_net:
#     i=0
#     while i<len(net):
#         print 'layr-'
#         print net[i]
#         layer=net[i]
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(cur_row)
            
            if act_func=="sigmoid":
                result=activate_sigmoid(sum)
            elif act_func=="relu":
                result=activate_ReLU(sum)
            elif act_func=="tanh":
                result=activate_tanh(sum)
                
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
            
        cur_row =prev_input
#         i+=1
    
    return cur_row


# In[540]:


def d_sigmoid(output):
    return output*(1.0-output)

# expand
def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


# In[541]:


def back_propagation(wgt_net,row,expected,act_func):
# while in place of for
    length=len(wgt_net)
    for i in reversed(range(length)):
        layer=wgt_net[i]
        layr_len=len(layer)
        errors=np.array([])
        if i==length-1:
            results=[neuron['result'] for neuron in layer]
            errors = expected-np.array(results) 
        else:
            for j in range(layr_len):
                herror=0
                nextlayer=wgt_net[i+1]
                for neuron in nextlayer:
                    herror+=(neuron['weights'][j]*neuron['delta'])
                errors=np.append(errors,[herror])

        for j in range(layr_len):
            neuron=layer[j]
            if act_func=="sigmoid":
                neuron['delta']=errors[j]*d_sigmoid(neuron['result'])
            elif act_func=="relu":
                neuron['delta']=errors[j]*d_ReLU(neuron['result'])
            elif act_func=="tanh":
                neuron['delta']=errors[j]*d_tanh(neuron['result'])


# In[542]:


def updateWeights(net,input_layr,lrate):
    
    for i in range(len(net)):
        inputs = input_layr
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]


# In[543]:


def evaluate(net,row,act_func):
    ans=forward_propagation(net,row,act_func)
    return ans

def new_weight(net,lrate,row):
    updateWeights(net,row,lrate)  


def back_prop(net,row,expected,act_func):
    back_propagation(net,row,expected,act_func)
    
def training(net, epochs,lrate,n_outputs,act_func,loss_func):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs=evaluate(net,row,act_func)
#             eval
#             outputs=forward_propagation(net,row)
            
            expected=[0.0 for i in range(n_outputs)]
            expected[y[i]]=1
            
            if loss_func=="MSE":
                sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
            else:
                sum_error=cross_entropy(outputs,expected)
                
            back_prop(net,row,expected,act_func)
            new_weight(net,lrate,row)
#             back_propagation(net,row,expected)
#             updateWeights(net,row,lrate)

        if epoch%100 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors


# In[544]:


act_func="sigmoid"
loss_func="MSE"
errors=training(net,1000, 0.05,2,act_func,loss_func)


# In[545]:


epochs=[0,1,2,3,4,5,6,7,8,9]
plt.plot(epochs,errors)
plt.xlabel("epochs in 10000's")
plt.ylabel('error')

plt.savefig('errors_wrt_sigmoid_mse_.png')
plt.show()


# In[ ]:


# validate


# In[ ]:


# Make a prediction with a network
def predict(network, row,act_func):
    outputs = forward_propagation(net, row,act_func)
    return outputs


# In[ ]:


act_func="sigmoid"
validate


# In[ ]:


pred_y=[]
for i in range(len(validate)):    
    pred=predict(net,x_validate[i],"tanh")
    output=np.argmax(pred)
    print 'pred-',pred
    pred_y.append(output)
    
    
cor=0
for i in range(len(validate)):
    if y_validate[i]==pred_y[i]:
        cor+=1

accuracy=cor*100/len(validate)
print 'pred_y=',pred_y
print 'accuracy=', accuracy


# In[ ]:


# Y_train


# In[ ]:


ans=[2,15,8,1,9]
print np.argmax(ans)


# In[ ]:





# In[ ]:




