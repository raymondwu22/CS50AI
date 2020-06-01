## Neural Networks
- Key properties of biological neural networks
    - neurons are connected to and receive electrical signals from other neurons.
    - neurons process input signals and can be activated. If activated, that neuron is then able to propagate further
    signals in the future.
- **artificial neural network** - mathematical model for learning inspired by biological neural networks.
    - model mathematical fn from inputs to outputs based on the structure and parameters of the network.
        - the structure of the network ultimately determines its function.
    - allows for learning the network's parameters based on data.
        - we want a model that is easy for us to write code that allows for the network to be able to figure out how to
        model the right mathematical fn, given a set of input data.
   - we will use _units_ inside of the neural network that are represented like a **node** in a **graph** connected 
   by **edges**.
    - think of this idea as mapping from inputs to outputs.
    - try to figure out how to solve a problem, model a mathematical fn
        - e.g. we have inputs (x1, x2) and given these inputs we have a task like predicting if it will rain.
            - given the variables x1 and x2, come up with a Boolean classification (rain vs. no rain).
        - in the **learning** lecture, we defined h(x1,x2), our hypothesis fn:
            - question then becomes, what does h() do in order to make that determination?
            - last lecture, we used a linear combination of the x1 and x2 input variables to determine the output:
                - h(x1, x2) = w0 + w1x1 + w2x2
                    - remember that each input variable is multiplied by a **weight**
                    - w0 is not multiplied by an input variable at all, only serving to move the fn up or down
                        - think of it as a weight multiplied by a dummy value, e.g. 1
                        - sometimes also called a **bias** that is added to the result as well
            - effectively, we need to decide and figure out what the value of the weights should be to determine what 
            value to multiple to our inputs for a result.
            - at the end, we needed to make a classification (rain vs. no rain), and to do that classification, we 
            needed a function to define a threshold.
                - we saw the **step function** (hard threshold):
                    - g(x) = 1 if x ≥, else 0
### Activation Functions
- **activation function** - a function that determines when it is that the output becomes 'active'. e.g. when it changes
 to a 1 instead of being a 0. takes the output of multiplying the weights together and adding the bias, and then 
figuring out the output.
- but if we did not want a purely binary classification, we could use a different fn (soft threshold) that allows for
in-between real number values.
    - e.g. **logistic sigmoid** function (S-shaped curve)
        - g(x) = e^x / (e^x + 1)
- another example is the **rectified linear unit(ReLU)**: g(x) = max(0,x)
- each activation function can be thought of as a fn that gets applied to the result of all of the computation:
    - h(x1,x2) = g(w0 + w1x1 + w2x2)
### Neural Network Structure
- `h(x1,x2) = g(w0 + w1x1 + w2x2)` is the model for one of the simplest neural networks we are going to represent 
graphically by using **nodes and edges**
    - input will be connected to the output via edges and the edges will be defined by their weights. 
    - the **output unit** will then calculate an output based on the inputs and the weights.
        - as a reminder, the output function, g(w0 + w1x1 + w2x2), will then be passed onto an activation fn.
- Example: recall the **OR** function for propositional logic
    - |  x  |  y  |f(x,y) |
      |-----|-----|-------|
      |  0  |  0  |   0   |
      |  0  |  1  |   1   |
      |  1  |  0  |   1   |
      |  1  |  1  |   1   |
    - How can we train a neural network to be able to learn this particular OR function? What would the weights look 
    like?
        - use 1 for the weights and a -1 for the bias.
        - activation function = g(-1 +1x1 + 1x2)
            - g(-1 +1(0) + 1(0)) = 0
            - g(-1 +1(1) + 1(0)) = 1
            - g(-1 +1(1) + 1(1)) = 1
- Example: **AND** function
    - |  x  |  y  |f(x,y) |
      |-----|-----|-------|
      |  0  |  0  |   0   |
      |  0  |  1  |   0   |
      |  1  |  0  |   0   |
      |  1  |  1  |   1   |
    - use 1 for the weights and a -2 for the bias.
    - activation function = g(-2 +1x1 + 1x2)
- imagine generalizing this idea to calculate more complex functions as well:
    - e.g. given humidity and pressure, the probability of rain.
    - e.g. regression style problem, given advertising for a month, predict expected sales amount.
- for many problems we will not just have two inputs, and one benefit of neural networks is the ability to compose 
multiple units together to make our networks more complex.
- We can easily add more inputs and corresponding weights as we make the network more complex
    -`g(Σxiwi+w0)`
- How do we go about training the internal networks? We would like our neural network to be able to calculate the 
weights so that our neural network can accurately model the function we want to estimate.
### Gradient descent
- **gradient descent** - algorithm for minimizing _loss_ when training a neural network.
    - loss refers to how bad the hypothesis function happens to be, how poorly does it model the data.
    - the loss function is just a mathematical function, and in calculus we can calculate the **gradient** or slope.
    The direction is the loss function is moving at any particular point.
        - Tells us which direction we should be moving the weights in order to minimize the amt of loss.
- high level idea for `gradient descent`:
    - Start with a random choice of weights.
    - Repeat:
        - Calculate the gradient based on **all** data points direction that will lead to decreasing loss.
        - Update weights according to the gradient.
- **Ask**: What is going to be the expensive part of doing the calculation? 
    - Having to take all the data points and using all of those data points, figure out what the gradient is at this 
    particular setting of all of the weights.
    - Will have to repeat this process many many times.
    - We will want to be able to train our neural networks faster to be able to more quickly converge to some solution
    that is ultimately a good solution to the problem.
- **stochastic gradient descent** - 
    - Start with a random choice of weights.
    - Repeat:
        - Calculate the gradient based on **one data point**; direction that will lead to decreasing loss.
        - Update weights according to the gradient.
    - Note that using one data point instead of all points will probably give a less accurate estimate of what the 
    gradient actually is, but we will be able to much more quickly calculate the gradient.
- **mini-batch gradient descent** - 
    - Start with a random choice of weights.
    - Repeat:
        - Calculate the gradient based on **one small batch**; direction that will lead to decreasing loss.
        - Update weights according to the gradient.
    - Note that using one data point instead of all points will probably give a less accurate estimate of what the 
    gradient actually is, but we will be able to much more quickly calculate the gradient.
- Supervised machine learning with neural networks - give neural network input data that corresponds to output data 
(labels).
    - using the data, the algorithm can use gradient descent to figure out the weights to create a model to predict 
    weather.
- Networks can also be structured to perform **multi-class classifications** where we map multiple input units to 
multiple output units. So far, we've only focused on one output, which works for binary classifications (e.g. rain vs.
no rain)
    - as we add more inputs or outputs, to be able to keep the network fully connected, we need to ad more weights so
    that each of the input nodes are connected to each output node.
    - e.g. in weather predicting, we don't only care if it is raining or not. There are multiple different categories
    e.g. sunny, cloudy, rainy, snowy
        - input variables would get multiplied by each of the various weights and after passing the result through a
        activation function in the outputs, we end up with a number (probability).
- Neural networks can also be applied to reinforcement learning.
    - train agent to learn what action to take depending on what state they are in.
    - input data will represent the state the agent is in.
    - output will be the different actions the agent can choose from.
    - based on inputs, we will calculate values for each output and the outputs could model which actions are better /
    which the AI should take.
- Neural networks are broadly applicable, so that anything we can frame as a mathematical function can be modeled by 
neural networks (gradient descent).
- How do we train neural networks with multiple outputs? Think of each output as a separate neural network.
    - additional step at the end to turn the values into a probability distribution.
- **Limitations of this approach**: can only predict things that are _linearly separable_
    - a single unit that is making a binary classification, or **perceptron**, can not deal with more complex situations
     where there is no straight line that can divide the data (e.g. a circle)
- **perceptron** - only capable of learning linearly separable decision boundary.
    - may not generalize well to situations that involve real work data that is not linearly separable.
### Multilayer Neural Networks
- **multilayer neural network** - artificial neural network with an input layer, an output layer, and at least one 
_hidden layer_.
    - hidden layer is separate from the I/O nodes, and calculates its own values as well.
        - inputs are no longer directly connected to the output. the hidden layer is a intermediary.
    - hidden layer will calculate its own output or activation, based on a linear combination of all the inputs
    - In effect, start with the inputs that are multiplied by weights to calculate values for the hidden nodes. The 
    hidden nodes are multiplied by weights in order to figure out the output.
- advantage of layering gives us the ability to model more complex functions. Instead of a single decision boundary/line
- each hidden node can learn a different decision boundary that is then combined to figure out the ultimate output.
- each hidden node can learn a useful property or feature of all the inputs and then learn how to combine the features
together to get the output.
- how do we train a neural network with hidden layers?
    - we are given the inputs and what the output should be.
    - the input data does not tell any information regarding what value the hidden node should be
### Backpropagation
- if you know what the error / loss is on the output node, you can back propagate the error for how much the error and
figure out what the error is from each of the nodes in the hidden layer. 
- **backpropagation** - algorithm for training neural networks with hidden layers.
    - Pseudocode:
        - start with a random choice of weights.
        - repeat:
            - calculate error for output layer.
            - for each layer, starting with output layer, and moving inwards towards earliest hidden layer:
                - propagate error back one layer.
                - update weights.
- backpropogation is what makes neural networks possible. It makes it possible to take multi-level structures and train 
those structures to update those weights and ultimately create a function that is able to minimize the total amt of 
loss.
- **deep neural networks** - neural network with multiple hidden layers.
    - this is what ultimately allows us to be able to model more sophisticated types of functions. Each layer can 
    calculate something a bit different and we can ultimately combine the info to figure out what the output should be.
### Overfitting
- As we begin to make our models more complex, we run the risk of overfitting our function to the training data.
- **overfitting** - a model that fits too closely to a particular data set and therefore may fail to generalize to 
future data.
- risk becoming over-reliant on certain nodes to calculate things based off of the input data and doesn't allow us to 
generalize for the output
    - popular technique to deal with this is called dropout
- **dropout** - temporarily removing units - selected at random - from a neural network to prevent over-reliance on 
certain units.
    - after the training process, hopefully end up with a network that is more robust and does not rely too heavily on
    one particular node. More generally learns how to approximate functions.
### Tensorflow
- one of the most popular machine-learning neural network libraries
- Keras is one of the APIs in tf.
- ReLU activation function, as a reminder: g(x) = max(0,x) 
- an "epoch" is equivalent to one loop
### Computer vision
- **computer vision** - computational methods for analyzing and understanding digital images.
    - e.g. social media sites can label/tag people in photos. self-driving cars. handwriting recognition
- neural networks rely on input (numerical data), many units with each one representing a number. In the context of 
an image or handwriting recognition, think of an image as a grid of pixels, or as a grid of dots where each dot has a 
color.
    - handwriting recognition: imagine filling in a grid in a particular way where a 2 or 8 is generated
    - each pixel value can be represented using numbers (0 = black, 255 = white)
         - color usually represented with 256 numbers, so colors can be represented using **8 bits**.
     - color image rgb(255,255,255)
- imagine a neural network that processes images, with one (black/white image) or three numbers (color images) that is 
connected to a deep neural network. The deep neural network may take all of the of the pixels of the handwriting image
and the output could be one of 10 neurons that classify it between 0-9
- drawbacks:
    - large size of input array and therefore large amount of weights that need to be calculated
    - by flattening everything into a structure of pixels, we lose access to information about the structure of the 
    image.
        - when a person looks at an image, they may look at the curves, shapes or things in different regions of the 
        image that may be combined to see a larger photo.
        - it can therefore be helpful to use properties about the image itself, such as the structure of the image, to
         improve about the way we learn based on the image.
 - in order to figure out how to train our neural networks to be able to better deal with images, we will look at 
 several algorithms that allow us to take the images and extract info out of the image.
 ### Image convolution
 - **image convolution** - applying a _filter_ that adds each pixel value of an image to its neighbors, weighted 
 according to a kernel matrix
    - can extract valuable information after applying a convolution filter. e.g. predict if there is a curve in the 
    image, if it forms the outline of a line or shape.
    - e.g. kernel matrix: [0,-1,0,-1,5,-1,0,-1,0]
        - in this example, you would evaluate 3x3 sections of a image and apply the filter to calculate a value for that 
        section.
    - e.g. kernel matrix: [-1,-1,-1,-1,8,-1,-1,-1,-1]
        - famous example. if all values in the 3x3 matrix are the same value then the result is 0. when the middle value
        is bigger than the neighboring values, will have a positive output.
        - this filter can be used to detect edges in an image (boundaries).
- although image convolution extracts details from the images, the images are still very big.
    - large images pose a few problems:
        - large inputs for the neural network
        - have to pay close attention to each particular pixel.
            - usually when we look at a image, we do not care whether something is in one particular pixel or 
            immediately next to it.
            - we only care if there is a feature in some region of the image.
- **pooling** - reducing the size of an input by sampling from regions in the input.
    - ultimately taking a big image and turn it into a smaller image.
- **max-pooling** - pooling by choosing the max value in each region.
    - e.g. starting with a 4x4 image, we want to reduce the dimensions of the image and make it a smaller image so 
    there are fewer inputs to work with. 
    - e.g. 2x2 max pool will look at 2x2 regions and extract the max value from that region.
    - we are then able to make our analysis independent of whether a particular value was in pixel a or b, as long as it
    was in the general region, we will have access to that value. makes the algorithm more robust.
### Convolutional Neural Networks
- **convolutional neural networks (CNN)** - neural networks that use convolution, usually for analyzing images.
- start w/ input image/grid of pixels -> **convolution** (apply multiple filters to get a **feature map**)
-> **pooling** -> **flattening** -> input into traditional neural network
    - each feature map may extract different important characteristics of the image
        - we can train neural networks to learn the weights between particular units, as well as to learn what these 
        filters should be, what type of filter will minimize the **loss function** and minimize how poorly our 
        hypothesis performs in figuring out the classification of an image for example.
    - pooling will further reduce the dimensions/size of the images (e.g. max-pooling or average-pooling)
        - makes it easier for us to deal with; fewer inputs
        - makes the algorithm more resilient/robust against potential movements of particular values by one pixel
    - a flattened image will include one input for each of the values of the resulting feature maps after we do the 
    convolution and pooling steps.
    - note that there is no reason necessarily to only apply each step once (convolution, pooling, flattening), in fact
    in practice, convolution and pooling can be used multiple times in multiple different steps (re-applied)
        - goal of this sort of model is that in each of these steps, we learn different types of features of the 
        original image
        - **first convolution and pooling step** - learn low-level features; e.g. edges, curves and shapes 
        - **second convolution and pooling step** - learn high-level features; e.g. objects, eyes, digits, etc.
- `softmax activation fn` takes the output and turns it into a probability distribution
- abstract representation of our neural network
    - input -> network -> output
        - input may be a vector of values
        - network does some calculations on the input and provides an output
    - also known as a **feed-forward neural network** 
### Recurrent Neural Networks
- **feed-forward neural network** - neural network that has connections only in one direction.
    - very helpful in solving classification problems
    - limitations to feed-forward neural networks.  Input needs to be of a fixed shape (fixed number of neurons in the 
    input). Output also needs to be of a fixed shape (fixed number of neurons in the output).
- **recurrent neural network** - neural network that generates outputs that are later fed back into the neural network
as input for future runs of that network.
    - input -> network <-> output
    - this ultimately allows the network to store **state**, some information that can be used for future runs of the 
    network.
    - particularly helpful when trying to deal with _sequences_ of data
        - e.g. Microsoft's implementation for `CaptionBot`
    - allows us to do a **one-to-many relationship** for inputs to outputs, whereas more 'vanilla' neural networks can
    only do a **one-to-one** neural network where one input is mapped to only one output.
    - the output will not represent a whole sequence of words, since we will just be using a fixed set of neurons. 
    Instead, the output will just be the first word. We train the network to output what the first word of the caption.
    The initial output will then be fed back in as new input data and that second pass through the neural network will
    then output the second word, etc.
        - where each pass through a neural network will represent one word at a time.
- There are other models or ways we can try and use recurrent neural networks.
    - e.g. Youtube needs to look at **videos** to detect copyright violations or categorize videos as educational, etc.
    - for videos, we can pass in each frame at a time, where we take the first frame to pass into the network, and then
    not output anything yet, but rather let the network take in another input, but this time, the network gets info 
    from the last time, etc. where each time the network gets the most recent input as well as information that was 
    processed from all of the previous iterations.
        - this is then a **many-to-one relationship** learning, where you take a **sequence** (video, message/text,
         spoken-language) and output 
            - e.g. audio message and voice recognition
- one last category: translation 
    - for Google translate, it takes in a sequence of words and outputs a sequence as well.
    - **many-to-many relationship**
- number of different types of recurrent neural networks: `Long-Short term memory (LSTM) neural networks`



        
    
    


