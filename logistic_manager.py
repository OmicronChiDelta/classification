import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


logit = lambda z: 1/(1+np.exp(-z))
loss = lambda p, y: -(y*np.log(p) + (1-y)*np.log(1-p))

##UNIT TEST: lambda function 
#z = np.linspace(-10, 10, 50)
#p = logit(z)
#plt.plot(z,p)
#plt.show()


#UNIT TEST: Dummy data
x = np.random.normal(-2, 3, 100)
x = np.concatenate((x, np.random.normal(2, 1, 100)), axis=0)
y = np.zeros(100,)
y = np.concatenate((y, np.ones(100,)), axis=0)
plt.scatter(x, y)
plt.show()

#Hack to get validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.5, shuffle=True)
#plt.scatter(x_train, y_train)
#plt.scatter(x_val, y_val)
#plt.scatter(x_test, y_test)
#plt.show()

#Append a trailing 1
x_train = np.array([[i, 1] for i in x_train])
x_val = np.array([[i, 1] for i in x_val])
x_test = np.array([[i, 1] for i in x_test])

#Hyperparameters
epoch_count = 50
grad_step = 0.01
batch_size = 30
num_feats = 1
num_examples = x_train.shape[0]
batch_indices = np.random.randint(0, num_examples, batch_size)
loss_total = 0
loss_history = []

#Initialise weights
#omega = np.random.normal(0, 1, num_feats+1)
omega = np.zeros(num_feats+1,)

for k in range(epoch_count):
    
    #Calculate validation error
    v_op = [logit(np.matmul(omega, i)) for i in x_val]
    v_pred = []
    
    for i in v_op:
        if i > 0.5:
            v_pred.extend([1])
        else:
            v_pred.extend([0])
            
    v_pred = np.array(v_pred)
    v_score = np.sum(v_pred == y_val)/x_val.shape[0]
    print(k, loss_total, v_score)
    
    #Sample a random batch of training data
    batch_indices = np.random.randint(0, num_examples, batch_size)
    x_batch = [x_train[i] for i in batch_indices]
    y_batch = [y_train[i] for i in batch_indices]
    
    #Compute forward pass
    p_batch = [logit(np.matmul(omega, i)) for i in x_batch]
    
    #Compute gradient (mean of batch gradients)
    grad_comps = [(-y_batch[i] + p_batch[i])*x_batch[i] for i in range(batch_size)]
    grad = np.mean(grad_comps, axis=0)
    
    #Compute loss function (mean of batch losses)
    loss_comps = [loss(p_batch[i], y_batch[i]) for i in range(batch_size)]
    loss_total = np.mean(loss_comps, axis=0)
    loss_history.extend([loss_total])
    
    #Update weights
    omega = omega - grad_step*grad
    
plt.plot(loss_history, 'ro-')
plt.show()