#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import math
eps = np.finfo(float).eps
from numpy import log2 as log
import matplotlib.pyplot as plt 


# In[52]:


XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
X_train=XORdata[:,0:2]
Y_train=XORdata[:,-1]


# In[53]:


def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


# In[54]:


def initialise_network(num_hidden_layers,num_hidden_nodes,class_labels_count):
    num_input_nodes = len(X_train[0])
    num_output_nodes = class_labels_count
    
    net = []
    h = 1
    hidden_layer = [ { 'weights': np.random.uniform(size=num_input_nodes)} for i in range(num_hidden_nodes) ]
    net.append(hidden_layer)
    while (h < num_hidden_layers) :
        num_hidden_nodes = len(net[-1])
        hidden_layer = [ { 'weights': np.random.uniform(size=num_input_nodes)} for i in range(num_hidden_nodes) ]
        net.append(hidden_layer)
    output_layer = [ { 'weights': np.random.uniform(size=num_hidden_nodes)} for i in range(num_output_nodes) ]
    net.append(output_layer)
    
    return net
        


# In[55]:


net=initialise_network(1,3,2)
# display(Image("img/network.jpg"))
print_network(net)


# In[56]:


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))


# In[57]:


def forward_propagation(net,input,activation_function):
    print 'going inside forwrd func'
    row=input
#     print 'row is :',row
    for layer in net:
        prev_input=np.array([])
#         print 'length of layer is :',len(layer)
        for neuron in layer:
#             print 'neuron is :',neuron
#             print 'neuron[weights] is :',neuron['weights']
#             print 'neuron[weights].T.shape is :',neuron['weights'].T.shape
#             print 'row.shape is :',row.shape
            sum=neuron['weights'].T.dot(row)
            
            if(activation_function == 'sigmoid'):
                result=activate_sigmoid(sum)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input
    print 'going out of forward func'
    return row


# In[58]:


def sigmoidDerivative(output):
    return output*(1.0-output)


# In[59]:


def back_propagation(net,row,expected,activation_function):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results) 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                if(activation_function == 'sigmoid'):
                    neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])


# In[60]:


def updateWeights(net,input,lrate):
    
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]


# In[61]:


def training(net, epochs,lrate,class_labels_count,cost_function,activation_function):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X_train):
            print 'i is :',i
            print 'row is :',row
            outputs=forward_propagation(net,row,activation_function)
            
            expected=[0.0 for i in range(class_labels_count)]
            expected[Y_train[i]]=1
    
            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
            back_propagation(net,row,expected,activation_function)
            updateWeights(net,row,0.05)
        if epoch%10000 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors


# In[ ]:


errors=training(net,100000, 0.05,2,'MSE','sigmoid')


# In[ ]:


epochs=[0,1,2,3,4,5,6,7,8,9]
plt.plot(epochs,errors)
plt.xlabel("epochs in 10000's")
plt.ylabel('error')
plt.show()


# In[ ]:




def predict(network, row):
    outputs = forward_propagation(net, row,'sigmoid')
    return outputs

prediction_list = []
pred=predict(net,np.array([1,0]))
print(pred)
output=np.argmax(pred)
prediction_list.append(output)

print(output)

correct = 0
for i in range(len(prediction_list)):
    if (prediction_list[i] == Y_train[i]):
        correct = correct+1

accuracy = (correct/len(Y_train))*100
print 'accuracy is :',accuracy
