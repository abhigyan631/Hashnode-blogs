---
title: "A Deep Dive into Optimizers: Adam, Adagrad, RMSProp, and Adadelta"
seoTitle: "Deep Dive into Optimizers like Adam, Adagrad, RMSProp, Adadelta"
datePublished: Mon Mar 27 2023 16:27:07 GMT+0000 (Coordinated Universal Time)
cuid: clfr1l0gc000709lg73e80fbc
slug: a-deep-dive-into-optimizers-adam-adagrad-rmsprop-and-adadelta
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/vpOeXr5wmR4/upload/b3a3d0a7c927d4f4d18e2fd67810dcab.jpeg
tags: python, data-science, neural-networks, deep-learning, jupyter-notebook

---

### **Introduction**

This article is **the next part of** [**this article**](https://hashnode.com/edit/clfp9pmyk000509jn0y2ab809) on **Optimization algorithms** where we discussed **Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent, and Momentum-based Gradient Descent Optimizers.** In this article, we will discuss the optimization algorithms like **Adagrad, Adadelta, RMSProp, and Adam**. We will provide the **definition, advantages and disadvantages**, **code snippets, and use cases** like the previous article.

1. ### **Adagrad**:
    
    This is an **adaptive learning rate optimization algorithm** that **adjusts the learning rate** for each parameter **based on its historical gradients**. The idea is to **increase the learning rate for parameters with small historical gradients** and **decrease it for those with large historical gradients**. The update rule is as follows:
    
    ```plaintext
    cache += gradient_of_loss ** 2
    w -= (learning_rate / (sqrt(cache) + epsilon)) * gradient_of_loss
    ```
    
    * `gradient_of_loss` is the **gradient of the loss function** with respect to the **weights at a particular iteration of the training process**. This gradient is **computed using backpropagation**.
        
    * `cache` is a **cache of historical squared gradients** for each weight parameter. The cache is initialized to zero at the start of training.
        
    * `learning_rate` is a hyperparameter that **controls the step size** for weight updates.
        
    * `epsilon` is a small value added to the denominator to prevent division by zero.
        
    
    **Advantages of Adagrad:**
    
    * Adagrad **adapts the learning rate** for each weight parameter based on the **historical gradient information**. This can be useful for problems **with sparse data** or when the **optimal learning rate** varies widely across different parameters.
        
    * Adagrad has been shown to **converge faster than standard gradient descent methods** in many cases.
        
    * Adagrad does **not require manual tuning** of the learning rate, which can be a **time-consuming process** in other optimization algorithms.
        
    
    **Disadvantages of Adagrad**:
    
    * Adagrad accumulates the **squared gradients** in the cache over time, which can lead to a very **small learning rate** later in the training process. This can **result in slow convergence** or even **stagnation in the optimization** process.
        
    * Adagrad is **not suitable for non-convex optimization problems** since it is possible for the **accumulated gradients to become very large** and cause the **learning rate to become very small**, preventing the algorithm from escaping local minima.
        
    
    **Code:**
    
    ```python
    from keras.optimizers import Adagrad
    
    # create a neural network model
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=100))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # compile the model with Adagrad optimizer
    optimizer = Adagrad(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    ```
    
    We begin by importing the Adagrad optimizer from the **Keras package** into this code. Next, using a **neural network model** with a single input layer, a **hidden layer with 32 units** and a **ReLU activation function**, and a **single output layer with a sigmoid activation function**, we achieve our goal. The binary **cross-entropy loss function** and the **Adagrad optimizer with a 0.01 learning rate** are used to construct the model.
    
    The **fit() function**, which accepts the **training data and labels**, number of **epochs, batch size, and validation data** as inputs, is then used to train the model.
    
    The **learning rate** is a parameter passed to the **Adagrad() method**, which specifies the **Adagrad optimizer**. The optimizer adjusts the learning rate for each weight parameter based on the **historical gradient data** and **updates the model's weights** during **training based on the gradients** of the **loss function** relative to the weights.
    
    **Use cases for Adagrad:**
    
    1. Adagrad can be used in problems with **sparse data** or in problems where the **optimal learning rate varies widely** across different parameters. This is because Adagrad **adapts the learning rate** for each **weight parameter based** on the **historical gradient information**, allowing it to adjust to the specific requirements of the problem.
        
    
    1. Adagrad can be used in **neural network training tasks where the manual tuning of the learning rate** can be a time-consuming process. Adagrad **eliminates the need for manual tuning**, making it a convenient choice for many neural network optimization problems.
        
    
    In general, Adagrad can be a useful optimization algorithm to try for neural network training tasks. However, it is **important to monitor the optimization process carefully to ensure that the learning rate does not become too small** and to switch to other optimization algorithms if necessary.
    

### **Adadelta:**

The Adadelta optimizer is an **extension of the Adagrad optimizer** that seeks to improve its drawbacks, mainly the **decrease in the learning rate** over time. The Adadelta optimizer replaces the learning rate with an **adaptive learning rate that varies with time** and a **weighted moving average of the gradients**.

1. The Adadelta optimizer updates the weight using the following formula:
    
    ```plaintext
    cache = decay_rate * cache + (1 - decay_rate) * gradient_of_loss ** 2
    delta = sqrt((delta_cache + epsilon) / (cache + epsilon)) * gradient_of_loss
    delta_cache = decay_rate * delta_cache + (1 - decay_rate) * delta ** 2
    w = w - delta
    ```
    
    The running averages of the **second moments of the gradient** and the **updates**, respectively, are **cache and delta\_cache** in this instance. The **decay** of these averages is controlled by the hyperparameter **decay\_rate**. A tiny constant called **epsilon** **prevents division by zero**.
    
    The update rule for the weight `w` involves a **weighted average of the gradients**, where the weights are given by the `delta` term. The `delta` **term** is calculated by **dividing the running average of the updates** by the running average of the second moments of the gradients.
    
    The **Adadelta optimizer** effectively scales the learning rate based on the **second moment of the gradients**, which can help to **improve the convergence of the optimization process**, especially in the presence of **noisy gradients or sparse data**.
    
    **Advantages:**
    
    * Adadelta **automatically adapts the learning rate**, eliminating the need to manually tune it.
        
    * Adadelta uses a **memory of past gradients to adjust the learning rate**, which can lead to **faster convergence and better generalization**.
        
    * Adadelta is **robust to noisy gradients** and **works well with sparse data**.
        
    
    **Disadvantages:**
    
    * Adadelta requires **more memory** than some other optimization methods, as it **maintains a running average of the second moments of the gradients**.
        
    * Adadelta may be **slower to converge** than some other optimization methods, especially on problems with smooth and well-behaved loss surfaces.
        
    
    **Code:**
    
    ```python
    # Initialize weights, decay rate, and epsilon
    w = np.zeros((n_features, 1))
    decay_rate = 0.9
    epsilon = 1e-6
    
    # Initialize running average of gradients and running average of updates
    grad_squared = np.zeros((n_features, 1))
    update_squared = np.zeros((n_features, 1))
    
    # Perform Adadelta optimization
    for i in range(n_iterations):
        # Compute gradient of loss function
        grad = compute_gradient(X, y, w)
        
        # Update running average of gradients
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad**2
        
        # Compute update
        update = np.sqrt(update_squared + epsilon) / np.sqrt(grad_squared + epsilon) * grad
        w -= update
        
        # Update running average of updates
        update_squared = decay_rate * update_squared + (1 - decay_rate) * update**2
    ```
    
    In this code, we initialize the weights `w`, the **decay rate** for the running averages `decay_rate`, and a small value `epsilon` to **prevent division by zero**. We also initialize the **running average of the squared gradients** `grad_squared` and the **running average of the squared updates** `update_squared` to zero.
    
    In each iteration of the optimization loop, we c**ompute the gradient of the loss function** using the `compute_gradient` function, **update the running average of the squared gradients**, and **compute the update using the current values** of `grad_squared` and `update_squared`. We then update the weights `w` using the update and update the running average of the squared updates.
    
    Adadelta adaptively adjusts the learning rate based on the running average of the squared gradients and updates, which allows it to **converge faster and avoid oscillations** in the optimization process.
    
    **Use cases:**
    
    * Adadelta is well-suited for **training deep neural networks with large datasets**, where tuning the **learning rate can be difficult** and where the **gradients may be noisy or sparse**.
        
    * Adadelta can be useful in **online learning scenarios**, where the **data arrives continuously** and the learning rate needs to be **adjusted dynamically**.
        
    * Adadelta is a good choice for **optimizing non-convex functions where the loss surface may be highly irregular**, as it is less prone to **getting stuck in local minima** compared to **some other optimization methods**.
        
    
    * Adadelta is well-suited for **training deep neural networks** with **large datasets**, where **tuning the learning rate can be difficult** and where the **gradients may be noisy or sparse**.
        
    * Adadelta can be **useful in online learning scenarios**, where the data arrives continuously and the learning rate needs to be adjusted dynamically.
        
    * Adadelta is a good choice for **optimizing non-convex functions** where the **loss surface may be highly irregular**, as it is less prone to getting stuck in **local minima** compared to **some other optimization methods**.
        
    
    ### **RMSProp**
    
    The RMSProp optimizer uses the following formula to compute the update for the weights:
    
    ```plaintext
    cache = decay_rate * cache + (1 - decay_rate) * gradient_of_loss ** 2
    update = learning_rate * gradient_of_loss / (sqrt(cache) + epsilon)
    weight = weight - update
    ```
    
    Here, `cache` is a moving average of the squared gradient, `decay_rate` is a hyperparameter that **controls the weighting of the current gradient versus the historical gradient values** in the cache, `epsilon` is a small **constant added for numerical stability**, `learning_rate` is the **step size**, `gradient_of_loss` is the gradient of the loss function with respect to the weight, and `weight` is the current weight value being updated. The `sqrt` function computes the **element-wise square root of the cache**. The idea behind the **RMSProp optimizer** is to **adapt the learning rate based on the magnitude of the recent gradients** so that the **learning rate is smaller for parameters with large gradients and larger for parameters with small gradients**. This is achieved by dividing the gradient by the root mean square (RMS) of the historical gradients in the cache.
    

**Advantages of RMSProp:**

* It adjusts the **learning rate** adaptively based on the g**radient of the current mini-batch**, which can help **accelerate convergence** and **improve generalization performance**.
    
* It divides the **learning rate by a running average** of the magnitudes of the past gradients, which can help **reduce the influence of noisy and sparse gradients**.
    
* It works well for a **wide range of neural network architectures** and **optimization problems**.
    

**Disadvantages of RMSProp:**

* It requires **tuning the hyperparameters** such as the **learning rate** and **decay rate** to achieve good performance.
    
* It may **converge to suboptimal solutions** in certain cases, especially when the **gradients are noisy** or the **loss function is non-convex**.
    
* It may exhibit **slower convergence** than some other optimization algorithms such as Adam.
    

**Code:**

```python
# Initialize weights, learning rate, decay rate, and epsilon
w = np.zeros((n_features, 1))
alpha = 0.01
decay_rate = 0.9
epsilon = 1e-8

# Initialize cache
cache = np.zeros((n_features, 1))

# Perform RMSProp gradient descent
for i in range(n_iterations):
    grad = compute_gradient(X, y, w)
    cache = decay_rate * cache + (1 - decay_rate) * grad**2
    w -= alpha * grad / (np.sqrt(cache) + epsilon)
```

The **weight vector** w, **learning rate** alpha, **decay rate** decay\_rate, and **epsilon value** epsilon are all initialized first in the code above. In order to **track the running average** of the squared gradients, we also initialize the **cache vector** cache.

The **decay\_rate** and the **squared gradient** are used to update the **cache vector** cache during **each iteration** of the loop, and the **current weight** vector w is used to **compute the gradient of the loss function**. Once the **gradient has been divided by the square root** of the **running average of the squared gradients plus a small constant epsilon**, the weight vector w is updated using the RMSProp update rule.

**Use cases of RMSProp:**

* RMSProp can be used for a wide range of **neural network architectures and optimization problems**, including **feedforward neural networks**, **convolutional neural networks**, and **recurrent neural networks**.
    
* It can be useful for problems with **large datasets and sparse gradients**, where it can help reduce the **influence of noisy and irrelevant gradients**.
    
* It can be used in conjunction with other techniques such as **weight decay and dropout to regularize the model** and **improve generalization performance**.
    

### **Adam:**

The Adam optimizer combines the ideas of momentum-based optimization and RMSProp. The update rule for Adam is as follows:

```plaintext
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * (gradient ** 2)
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
weight_update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

* `m` is the moving average of the gradient,
    
* `v` is the moving average of the squared gradient,
    
* `m_hat` and `v_hat` are **bias-corrected estimates** of `m` and `v`, respectively
    
* `beta1` and `beta2` are the **exponential decay rates** for the moving averages,
    
* `t` is the current iteration number,
    
* `epsilon` is a small constant to avoid division by zero.
    

The **Adam optimizer** calculates a different learning rate for each parameter by taking into account the **historical first and second moments** of the gradients. The parameter update is **scaled by a factor based on the ratio of the root mean square** of the **second moment and the first moment of the gradients**. This normalization helps the optimizer work well even for ill-conditioned problems and **makes it less sensitive to the choice of hyperparameters**.

**Advantages of the Adam optimizer are:**

* The **adaptive learning rate** approach enables it to **converge quickly** and reliably, even for **ill-conditioned problems and noisy gradients**.
    
* It works well for a **wide range of neural network architectures and optimization tasks**, including **deep neural networks, recurrent neural networks, and generative models**.
    
* The **bias correction** of the moving averages allows the optimizer to work well even in the early stages of training when the estimates of the first and second moments are unreliable.
    

**Disadvantages of the Adam optimizer are:**

* The adaptive learning rate can sometimes cause the optimizer to **overshoot the minimum and oscillate around it**, leading to **unstable convergence or poor generalization performance**.
    
* The additional hyperparameters, such as the exponential decay rates and the epsilon value, need to be carefully tuned to achieve good performance. Improper hyperparameter settings can lead to **slow convergence or divergent behavior**.
    

**Code:**

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

# Generate some dummy data for binary classification
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=(1000, 1))

# Define the neural network model
model = Sequential()
model.add(Dense(32, input_shape=(10,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the Adam optimizer and binary crossentropy loss
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the dummy data
model.fit(X, y, epochs=10, batch_size=32)
```

In this code, we first generate some **dummy data for binary classification using NumPy**. We then define a simple neural network model with one **hidden layer of 32 neurons** and a **sigmoid output layer for binary classification**.

After that, the **model is built using the Adam optimizer** with the following parameters: a **learning rate of 0.001**, a **first-moment decay rate** (beta 1) of **0.9**, a **second-moment decay rate** (beta 2) of **0.999**, and a **minimal epsilon value of 1e-07** for numerical stability. Additionally, we define the binary **crossentropy loss as the objective function** and monitor classification **accuracy as a parameter throughout training**.

Finally, we **train the model on the dummy data** using the **fit() method** of the **Keras model class**, specifying the **number of epochs and batch size** for the training process.

**Use Cases of the Adam Optimizer**

The **Adam optimizer** is widely used in neural network training due to its **fast convergence, robustness, and adaptivity**. It is particularly useful for **large-scale problems with complex loss landscapes and noisy gradients**, where it can often **outperform other optimization algorithms like SGD and RMSProp**.

### **Conclusion:**

In conclusion, selecting the right optimizer is crucial for the training of neural networks. Each optimizer has its strengths and weaknesses and is suitable for different scenarios. **Adam, Adagrad, RMSProp, and Adadelta** are some of the most widely used optimizers in deep learning. Understanding their **mathematical formulations and implementation details can help in choosing the right optimize**r for the task at hand. While there is **no one-size-fits-all solution**, being **familiar with these optimizers can go a long way in improving the performance of neural networks**.