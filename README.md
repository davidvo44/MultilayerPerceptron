# multilayer-perceptron


http://neuralnetworksanddeeplearning.com/chap1.html

https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3

https://trainingnns.github.io/?utm_source=chatgpt.com#math_background

https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


### Perceptrons

Un type particuler de neuronne artificiel:
 Il prend plusieurs input et sort un seul output. Pour calculer output, on introduit le poid.( w1, w2, ...); Ce soint des nombre exprimant l'importance de ces inputs par rapport au output.
 Le resultat des neuronnes, soit 0 ou 1 est determine si la somme des poid exprime en  ∑j (wjxj) est moins ou pus que le seuil de resultat. If > seul = 1, else = 0

Bien sur, le perceptrons a plusieurs layer, la premiere layer correspond souvent a l'input et la derniere est l'output. Mais entre ces layers peuvent contenir des layer cache. Et le nombre de ces couches augmente le niveau de decision en rendent plus complexe et abstrait.




### softmax function
- Softmax Activation Function transforms a vector of numbers into a probability distribution, where each value represents the likelihood of a particular class.
- Each output value lies between 0 and 1.
- The sum of all output values equals 1.

For a given vector, z=[z1,z2,…,zn]z=[z1​,z2​,…,zn​]the Softmax function is defined as: 

### σ(zi​)= (e^zj) / ∑j=1n​ezj​ezi​​


## Gradient descent

Definition: Let’s kick things off with the basics. Imagine you’re standing on top of a hill, and your goal is to reach the lowest point in the valley below. The most efficient way to do this is to take small steps downhill, always moving in the direction that decreases your altitude the fastest.

That’s essentially what gradient descent does, but instead of finding the lowest point in a valley, it finds the minimum of a loss function in a machine learning model.

In more technical terms, gradient descent is an optimization algorithm that helps your model find the optimal set of parameters (like weights in a neural network) that minimize the loss function.

The loss function measures how far off your model’s predictions are from the actual values, so minimizing it is key to improving your model’s performance.

Types of Gradient Descent: Now, you might be wondering, “Are there different ways to make this journey downhill?” Absolutely! Gradient descent comes in a few different flavors, each with its own strategy for updating the model’s parameters:

- Batch Gradient Descent: This approach calculates the gradient of the loss function using the entire dataset. It’s like gathering all the information before taking a single, well-calculated step. While it’s accurate, it can be slow and computationally expensive, especially with large datasets.
    
- Stochastic Gradient Descent (SGD): Instead of using the entire dataset, SGD updates the model’s parameters using just one data point at a time. Imagine taking a step every time you see a small change in your surroundings — fast, but sometimes a bit erratic. This method is quicker but can lead to more fluctuations in the path to the minimum.
    
- Mini-Batch Gradient Descent: This is the Goldilocks option — not too big, not too small. Mini-batch gradient descent splits the dataset into small batches, updating the parameters after each batch. It balances the accuracy of batch gradient descent with the speed of SGD, making it a popular choice in practice.

## BackPropagation

Backpropagation is the method by which your network “learns” from its mistakes. It calculates how the error should be distributed across the network’s weights so that, with each iteration, the model becomes more accurate.

Step-by-Step Process: Now, let’s break down how this all happens step by step:
- Forward Pass: Picture this: Your neural network is like a pipeline. In the forward pass, the input data flows through this pipeline — from the input layer, through the hidden layers, and finally to the output layer. The network makes a prediction based on the current weights.

- Loss Calculation: Once the network has made its prediction, it’s time to see how well it did. This is where the loss function comes into play. The loss function calculates the difference between the network’s prediction and the actual target value. The bigger the difference, the bigger the error.

- Backward Pass (Backpropagation): Here’s where the magic happens. Backpropagation starts at the output layer and works its way backward through the network, calculating the gradient of the loss with respect to each weight by applying the chain rule. It’s like retracing your steps to figure out where things went wrong. This backward pass assigns blame for the error to each weight, providing the information needed to correct the network’s course.

- Weight Update: Armed with these gradients, gradient descent then updates the weights in the direction that reduces the error. This process is repeated for many iterations until the network’s predictions are as accurate as possible.

## Relationship Backpropagation and Gradient Decent

Complementary Processes: Imagine you’re trying to sculpt a statue from a block of marble. Backpropagation is like the sculptor’s chisel — it figures out where to chip away based on the shape you want.

Gradient descent, on the other hand, is the force you apply with each stroke, gradually revealing the final form. These two processes are inherently complementary. Backpropagation computes the gradients, which tell you how much each weight in your neural network needs to be adjusted.

Gradient descent then takes those gradients and applies them, updating the weights to minimize the loss function. Without backpropagation, gradient descent wouldn’t know which direction to move in. And without gradient descent, backpropagation’s calculations wouldn’t result in any actual learning.

Workflow Integration: You might be wondering, “How do these processes fit together in a neural network’s workflow?” Here’s the typical sequence of events:

- Forward Pass: The journey begins with the forward pass, where your input data travels through the network. Each layer processes the data, ultimately producing an output — this could be anything from predicting a price to classifying an image.

- Loss Calculation: Once the network spits out a prediction, it’s time to measure its accuracy. This is done by comparing the prediction to the actual target using a loss function. The result is a single value that tells you how far off the mark the network was.

- Backpropagation (to Compute Gradients): Now, backpropagation kicks in. It starts at the output layer and works its way backward through the network, calculating the gradient of the loss with respect to each weight. This gradient is essentially a signal that tells each weight how much it contributed to the error.

- Gradient Descent (to Update Weights): Finally, gradient descent steps in to adjust the weights. Using the gradients computed by backpropagation, it updates each weight in the network, nudging them in the direction that reduces the loss. This step is like the network learning from its mistakes, gradually improving its performance over time.

## Adam

Adam (Adaptive Moment Estimation) is a optimisation algorithm that builds upon the strenghts of AdaGrad and RMSProp.

AdaGrad : Adaptive Gradiant adapt learning of each parameter by their historic
result = result - n * (g / (G + e))

g = actual gradiant
G = sum of (past gr)^2
e = const to prevent division by 0
n = learning Rate

Problem: if gradiant too big, or too small, can create problem




RMSProp:
Introduced to solve the AdaGrad problem. Instead of accumulating all past squared gradients as AdaGrad does, RMSProp uses a moving average.

v(t) = B * v(t - 1) + (1 - B) * g^2

v(t) = variance, accumulated moving average of squared gradients at time ttt
B = decay rate, around 0.9 or 0.95
gt = gradient at time ttt

update:
result = result - n * (g / (v(t) + e))


ADam:
Use m -> Mean of gradiant (direction)
and v -> MS square of mean of gradiant ( speed)

apply m and v on all weight and bias

result = result - n * (momentum / (sqrt(variance) + e))

mt = B1 * momentum + ( 1 - B1)* g
vt = B2 * variance + (1 - B2) * g^2

momentum = mt / (1 - B1 ** t)
variance = vt / (1 - B2 ** t)