#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os


# In[ ]:


#creating dataset
face_cascade = cv2.CascadeClassifier('C:\\Users\\snsha\\Desktop\\openCV\\Face Haarcascade Classifier.xml')
cap = cv2.VideoCapture(0)
face_id = input("Enter id and press enter")
#directory = 'C:\\Users\\snsha\\Desktop\\openCV\\dataset'
count = 0
while True:
   
    check, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        count += 1
        cv2.imwrite("dataset/User" + str(face_id) + '-' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow("capture",frame)
        
    key = cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break
    elif count >= 80:
        break
#print(os.listdir(directory)) 
cap.release()
cv2.destroyAllWindows()
        
         
        


# In[ ]:





# In[2]:


#required libraries
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[3]:


#importing dataset 
datadir = "C:\\Users\\snsha\\Desktop\\openCV\\dataset"
cat = ["Shashank","Nidhi"]
training_data = []
def create_data():
    for cate in cat:
        path = os.path.join(datadir,cate) #creating path to the image folder
        class_num = cat.index(cate)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img)) #full path to the image
            new_array = cv2.resize(img_array,(50,50))
            training_data.append([new_array,class_num])
create_data()
random.shuffle(training_data) # shuffling the dataset 
X = []
Y = []
for features,labels in training_data:
    X.append(features)
    Y.append(labels)
#saving the dataset as pickle file 
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
#slpitting images into test and train sets
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(X, Y, test_size=0.15, random_state=1)

x_train = np.array(x_train_orig)
x_test = np.array(x_test_orig)
y_train = np.array(y_train_orig)
y_test = np.array(y_test_orig)



            
        


# In[21]:




#method to read the dataset
#X = pickle.load(open("X.pickle","rb"))
#print(X[0])
#Y = pickle.load(open("Y.pickle","rb"))
#print(Y)



# In[5]:


#one hot encoding
def convert_to_one_hot(labels, C):
    C = tf.constant(C, name = "C")
    one_hot_matrix = tf.one_hot(labels,C,axis = 0)
    #sess = tf.compat.v1.Session
    #one_hot = sess.run(one_hot_matrix)
    #sess.close()
    return one_hot_matrix


# In[6]:


#flattening the images of trainig and test set
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
x_test_flatten = x_test.reshape(x_test.shape[0], -1).T
#normalizing image vectors
x_train = x_train_flatten/255.
x_test = x_test_flatten/255.
# Converting training and test labels to one hot matrices

y_train = convert_to_one_hot(y_train, 2)
y_test = convert_to_one_hot(y_test, 2)






print ("number of training examples = " + str(x_train.shape[1]))
print ("number of test examples = " + str(x_test.shape[1]))
print ("X_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))









# In[20]:


#creating tensors for x and y
x_train_tensor = tf.convert_to_tensor(x_train,dtype = tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train,dtype = tf.float32)
x_test_tensor = tf.convert_to_tensor(x_test,dtype = tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test,dtype = tf.float32)


    
    


# In[19]:


#initializing the parameters
def initialize_parameters():
    parameters = {}
    initializer_W = tf.initializers.GlorotUniform()
    initializer_b = tf.zeros_initializer()
    
    
    
    W1 = tf.Variable(initializer_W((50,7500)))
    b1 = tf.Variable(initializer_b(shape=[50,1], dtype=tf.float32))
    W2 = tf.Variable(initializer_W((50,50)))
    b2 = tf.Variable(initializer_b(shape=[50,1], dtype=tf.float32))
    W3 = tf.Variable(initializer_W((2,50)))
    b3 = tf.Variable(initializer_b(shape=[2,1], dtype=tf.float32))
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
    return parameters
parameters = initialize_parameters()



    
    
    
                  
                  
                  
                  
    
    
    

    
                  
                  
                  
                  
                  


# In[9]:


#forward propagation
parameters = initialize_parameters()
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X),b1)                                              
    A1 = tf.nn.relu(Z1)                                             
    Z2 = tf.add(tf.matmul(W2, A1),b2)                                            
    A2 = tf.nn.relu(Z2)                                            
    Z3 = tf.add(tf.matmul(W3, A2),b3)
    
    return Z3




# In[10]:


parameters = initialize_parameters()
z3 = forward_propagation(x_train_tensor, parameters)
print(z3)


# In[14]:


#computing cost 
def compute_cost(Z3, Y):
     
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels ))
   
    return cost
    


# In[17]:


#defining model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,num_passes = 1500,print_cost = True):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    #initializing parameters
    parameters = initialize_parameters()
    for i in range(num_passes):
        #forward propagation
        Z3 = forward_propagation(X_train, parameters)
        #computing cost 
        cost = compute_cost(Z3, Y_train)
        #backprop/gradient descent with adam optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        #printing cost after 100th iteraton
        if print_cost == True and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost == True and i % 100 == 0:
            costs.append(cost)
    #plotting the costs graph
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters 
    
    
    


# In[18]:


parameters = model(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)


# In[ ]:





# In[ ]:





# In[ ]:




