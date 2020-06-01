## Learning
- **Supervised Learning** - given a data set of input-output pairs, learn a function to map inputs to outputs
    - the computer trains its model on that given data set and begins to understand how the I and O are related
- **classification** is a task within supervised learning that involves learning a function mapping an input point to a 
discrete category
    - e.g. given banknotes, and need to categorize if it is `authentic` or `counterfeit`
    - e.g. predict weather given access to only historical data and have the computer look for patterns
        - Date, humidity (relative humidity), pressure (sea level, mb), rain
        - Will be supervised learning, because the dataset was labelled by a human to mark if it rained that day
            - to put this example more mathematecally, think of it as a function
                - f(humidity, pressure) => output is the category/label associated with the input
                - we would like to approximate this f() function
                - we will attempt to estimate the f() function with a hypothesis: `h(humidity, pressure)`
        - Where can we start? reasonable to start plotting these numerical values on a graph (can scale with more inputs 
        and multiple dimensions) 
            - it isn't unreasonable for computers to think in ten or hundreds of dimensions to be able to try to solve a 
            problem
            - plot data that corresponds to rainy days, and non-rainy days
            - goal: train a model so that if it is presented with new/unlabeled data, it will be able to predict what 
            classification should be used.
- **nearest-neighbor classification** - algorithm that, given an input, chooses the class of the nearest data 
point to that input.
    - things get tricky when the point is nearest to an outlier and can be classified incorrectly. It can 
    be helpful to look at the bigger picture, and there is a potentially an argument to be made to consider 
    the category of the majority of the neighbors.
- **k-nearest-neighbor classification** - algorithm that, given an input, chooses the most common class out 
of the **k** nearest data points to that input.
    - drawbacks: could be slow to measure the distance between points in the naive approach. 
        - note that there are ways around this, particular data structures that may improve the efficiency.
        There are also techniques that may be used to prune some of the data and remove the less relevant
        data points.
    - note that not every model is going to work under every situation. There are a number of different approaches 
    or algorithms to machine learning
        - each algorithm has trade offs, that depending on the data, one type of algorithm may be better at modeling
        for that situation than another algorithm.
    - We can use a **decision boundary** as another way to look at this classification, that separates the rainy days
    from the non-rainy days.
        - in the case of 2 dimensions, we can draw a line. Find some line, a separator, that divides the rainy from 
        the non-rainy days.
        - also known as **linear regression** to find a line that will separate the two halves from each other.
        - some times it may be possible to exactly separate the two halves, but realistically, data is messy and 
        there are outliers or noise in the data. Therefore, the data may not always be **linearly separable**, but a 
        line can still do a 'pretty good job' of separating and making that prediction.
        - to formalize our definition mathematically:
            - **inputs** to machine learning algorithm:
                - x1 = Humidity
                - x2 = Pressure
            - **output** by the hypothesis function:
                - h(x1,x2)= measure which side of the boundary this data point is on
                    - h(x1,x2) = Rain if `w0 + w1x1 + w2x2 ≥ 0`
                    - h(x1,x2) = No Rain otherwise.
            - **boundary**: a linear combination of the input variables (Humidity & Pressure)
                - take each of these inputs and multiply them by a number, called a **weight**, which measures the 
                importance of the numbers in determining the answer. A constant may be added as well to make the fn
                a little bit different. The result is then compared, is it greater than 0 or less than 0 to measure
                which side of the line the point belongs.
            - the expression, w0 + w1x1 + w2x2 ≥ 0, will determine if it the data point will be categorized as rain 
            or not
                - the expression will be a line, and the slope of the line depends upon the weights (w1 and w2).
                - goal: figure out what the weights will be so that it can accurately predict the classification of
                the points.
            - for the hypothesis function, we can sometimes use the names of the categories, but mathematically, if
            we are trying to make comparisons between these, it's easier to work with numbers:
                - h(x1,x2) = 1 if `w0 + w1x1 + w2x2 ≥ 0`
                - h(x1,x2) = 0 otherwise.
            - we will ultimately express this type of expression in a `vector expression` (e.g. list or tuple)
                - weight vector **w**: (w0, w1, w2)
                    - need the ML algorithm to determine what the weights should be.
                - input vector **x**: (1, x1, x2)
                    - this represents the data point that we are trying to predict a category for.
            - `w0 + w1x1 + w2x2 ≥ 0` is also known as the **dot product** of the two vectors, or simply put, taking 
            each of the terms of the two vectors and multiplying them together.
                - **w⋅x**: w0 + w1x1 + w2x2
            - can often see the hypothesis function written as:
                - hw(x) = 1 if **w⋅x** ≥ 0; 0 otherwise
                    - hypothesis fn can be said to be _parameterized by the weights_
                        - depending on what weights are chosen, we will end up getting a different hypothesis
                        - if we choose the weights correctly, we can do a good job estimating the outputs  
- **perceptron learning rule** - given a data point (x,y), update each weight according to:
    - wi = wi + a(y - hw(x))*xi
        - big picture idea: can start with random weights, but then learn from the data and shift the weights in 
        order to create a weight vector that is able to correctly more accurately estimate what the output 
        should be
    - can also be expressed as:
        wi = wi + a(actual value - estimate)*xi
            - if correct prediction, actual and estimate will be equal and the difference is 0
            - if incorrect, we need to make some changes to the weight to better predict the data point:
                - if actual > estimate, then we need to increase the weight to increase the output and make it 
                more likely to get the actual right value
                - likewise, if actual < estimate, then we need to decrease our weight to improve the output
            - the value alpha, or the learning rate, is a parameter or number that we choose for how quickly we 
            update the weight values. 
                - if alpha is bigger, we will update the weight values by a lot, and vice versa.
    - we ultimately end up with a **threshold function**
        - our hypothesis function will calculate the dot function, and if the output is greater than a 
        threshold value, then we declare it is a rainy day, otherwise not rainy.
            - this **hard threshold** only leaves two possible outcomes, and it leaves no room for confidence
            or strength of a prediction.
                - these functions are difficult to deal with and as you get deeper into ML, and want to take 
                derivatives of these curves, this type of functions are challenging
                - no notions of gradation or confidence
        - we can take advantage of **logistic regression**, or a logistic function or **soft threshold**
            - with this, we expand the number of possible values to any real number between 0 and 1.
                - e.g. a value of 0.7 would indicate that we are pretty confident it may rain, but again, we are
                 not as confident as some of other our other data points.
             - advantage: allows us to have an output that could be some real number that reflects a probability
             or likelihood that we think this particular data point belongs to a category.
 - **support vector machines** will try to find the maximum margin separator by finding the support vectors.
    - there are a lot of lines or decision boundaries that we can draw to separate two groups
    - goal: to select a line that is as far apart as possible from the 'red data' and 'blue data' categories 
    so that if we generalize a little bit and assume that maybe we have some points that are different from the 
    input and slightly further away, predict with some accuracy as to what that data point belongs to
    - **maximum margin separator** - boundary that maximizes the distance between any of the data points
    - **support vectors** are the vectors closest to the line and try to maximize the distance between the line and
    those particular points.
        - works the same way in 2D or higher dimensions (hyperplane)
             - the ability to work in higher dimensions can be helpful to deal with those cases that are not 
             linearly separable, or some data sets that don't have a line that divides the data.
             - e.g. can find a better fit for the data like a circle
 ### Regression
 - Classification is just one of the problems we may encounter in machine learning, where we are trying to predict a 
 discrete category. Red or blue, rain or not rain, authentic or category.
 - In other problems, we may want to predict a real number value.
 - **regression** - supervised learning task of learning a function mapping an input point to a continuous value
    - e.g. a company who wants to predict sales as a function of advertising spend.
        - f(advertising)
            - f(1200) = 5800
            - f(2800) = 13400
            - f(1800) = 8400
        - h(advertising)
            - hypothesis function that takes in a proposed advertising budget and predicts total sales from that 
            campaign
        - can use a linear regression approach to plot the data and draws a line that estimates the relationship between
        advertising and sales
            - we are trying to find a line that approximates the relationship, rather than separating two categories
### Evaluation Hypotheses
- With all these different approaches to solve ML style problems, how do we evaluate these approaches? How do we 
evaluate the various different hypotheses that we could come up with?
    - each of these algorithms will give us some sort of hypothesis, a function that maps inputs to outputs. We want to 
    know how well these functions work.
- can be thought of as an optimization problem
    - as a reminder, optimization functions either tr to maximize an **objective fn** by trying to find a global maximum
    - or minimize a **cost function**, by trying to find a global minimum
- **loss function** - fn that expresses how poorly our hypothesis performs
    - e.g. a loss of utility by whenever we predict something that is wrong, that adds to the output of our loss 
    function
    - a mathematical way to estimate numerical loss given our estimate and the actual output
    - some popular loss functions:
        - for discrete categories: **0-1 loss function**
            - L(actual, predicted)=
                - 0 if actual = predicted
                - 1 otherwise
            - goal is to come up with some hypothesis that minimizes the total empirical loss
                - we could take each of the input data point (each of which has a label) and compare it to the 
                prediction and assign it a numerical value as a result.
            - add up all of those losses across all of our data points to get an _empirical loss_.
    - other forms of loss that work better as we deal with more real value cases. 
        - e.g. cases where we map between advertising budget and amount that we do in sales. 
        - in these cases, we don't care if we get the values exactly right, but rather how close we are to the actual 
        value.
        - loss function should be able to take into account not just whether the actual = expected value, but also how
        far apart they are.
    - **L1 loss function**
        - `L(actual, predicted)= |actual - predicted|`
        - we can measure how far apart are the actual and predicted values
    - **L2 loss function**
        - `L(actual, predicted)= (actual - predicted)**2`
        - This version effectively penalizes much more harshly anything that is a worse prediction.
        - effective for minimiing the error on more outlier cases.
### Overfitting
- We run the risk of `overfitting` the line with any of these loss functions
- **overfitting** - a model that fits too closely to a particular data set and therefore may fail to generalize to 
future data
    - can happen in both the classification and regression cases
- Examine what we are optimizing for. In an optimization problem, all we do is say that there is some cost, and I want 
to minimize that cost.
    - Initially: cost(h) = loss(h)
        - can lead to overfitting since we are trying to find a way to perfectly match all of the input data
    - Modify the function by adding an additional variable: `cost(h) = loss(h) + λ*complexity(h)`
        - gives preference to a simpler decision boundary. Generally, a simpler solution is probably better and more 
        likely to generalize well to other inputs.
        - λ - e.g. if lambda is a greater value, we really penalize complex hypotheses.
### Regularization
- **regularization** - penalizing hypotheses that are more complex to favor simpler, more general hypotheses
    - `cost(h) = loss(h) + λ*complexity(h)`
- another way to prevent overfitting is to run experiments to see whether we cam generalize our model to other data sets
- oftentimes, we do not train our model with all of our data, but we employ a method called `holdout cross-validation`
- **holdout cross-validation** - splitting data into a training set and a test set, such that learning happens on the 
training set and is evaluated on the test set.
    - one downside of holdout cross-validation is the fact that we leave out a large amount of data that could have been
     used to train our model for validation.
- **k-fold cross-validation** - splitting data into k sets, and experimenting k times, using each set as a test set 
once, and using teh remaining data as training sets. 
    - e.g. divide into 10 different sets and run 10 different experiments. Each time for each of the 10 experiments, 
    can hold out one of those sets of data. Train on 9 sets, and test to see how well it predicts on set #10. Then pick
    another set of nine and test it on the other one that was held out. 
    - End up with 10 different results, 10 different answers for how accurately the model worked and can take the avg
    of the 10 to get an approx. on how well the model performs overall.
### Scikit-learn
- this python package has already written algorithms for nearest neighbor classification, perceptron learning, etc.
### Reinforcement Learning
- **reinforcement learning** - given a set of rewards or punishments, learn what actions to take in the future.
    - this type of learning involves learning from experience.
    - start an agent (AI or robot) that is situated in an environment where they will make their actions and
    ultimately receive a reward or punishment.
        - `environment` starts off by giving the `agent` a `state`.
        - in that `state`, the `agent` needs to choose an `action`.
        - after taking an `action`, the agent receives a new `state` and a `numerical reward` (+ meaning good;
         - meaning bad).
- In order to begin to formalize this, we need to first formalize the notion about states, actions and rewards as a
**Markov Decision Process**
 ### Markov Decision Processes
 - **Markov Decision Process** - model for decision-making, representing states, actions and their rewards.
 - recall `Markov Chains` have a bunch of individual states, and each state immediately transitioned to another state
 based on a probability distribution (recall the weather example from the uncertainty lecture).
 - imagine generalizing this idea where we just have states, where one state leads to another state according to 
 a probability distribution. 
    - Markov chains do not involve an agent, but it was entirely probability-based where there was a probability that
    transitions to one state vs. another.
- we not have an agent that is able to choose from a set of actions, where they have multiple paths forward that each
lead down different paths.
- we will add another extension, where anytime you move from a state taking an action, going into another state, we 
can associate a reward with that outcome.
    - `r` is positive (positive reward) or negative (punishment)
- Can define Markov Decision Processes:
    - Set of states `S`
    - Set of actions `ACTIONS(s)`
    - Transition model `P(s'|s,a)`
         - given state `s` and taking action `a`, what is the probability that we end up in state `s'`
         - can be deterministic (always end up with a new state given an action)
         - can also be probabilistic, where there is some randomness to in the world that can affect whether or not
         we end up in the exact same state given one action.
     - Reward function `R(s,a,s')`
 - Example: we have an agent on a map with a goal to reach an area (endpoint), and there are restricted areas that 
 are initially unknown to that agent that punish the agent if they land there.
    - if the agent just takes an action that ends up with punishments, it can learn that when it is in this state in the
    future, don't take the action: 'move to the right' from that initial state.
    - The agent learns over time which actions are good in particular states, and also which actions are bad. It can 
    follow the experience that it has learned over time.
### Q-learning
- **Q-learning** - method for learning a function Q(s, a), estimate the value of performing action `a` in state `s`.
    - originally we won't know what this Q-function will be, but over time, based on experience and trial/error, we 
    can learn what Q of s/a is for any particular state and action we may take in that state.
    - Approach:
        - State with `Q(s,a)` = 0 for all `s,a`
            - As i interact with the world and experience rewards or punishments, I want to update tmy estimate of 
            `Q(s,a)`.
        - When we take an action and receive a reward:
            - Estimate the value of `Q(s,a)` based on current reward and expected future rewards.
            - Update `Q(s,a)` to take into account old estimate as well as our new estimate.
    - Formally:
        - State with `Q(s,a)` = 0 for all `s,a`
        - Every time we take an action `a` in state `s` and observe a reward `r`, we update:
            - `Q(s,a) <- Q(s,a)+a(new value estimate - old value estimate)`
                 - we will need to decide how much we want to adjust our current expectation of what the value is of 
                 taking the action at that particular state.
                 - what the difference is determines how much we add or subtract from the existing notion of how much
                 we expect the value to be, is dependent on the parameter `a` or the learning rate.
                    - alpha represents how much we value new info compared to old info
                    - alpha value of 1 means we really value new info. If we have a new estimate, it does not matter 
                    what the old estimate is and we will only consider the new estimate
                    - 0 would mean ignore all the new info and keep Q the same.
                - over time as we go through a lot of experiences, we already have some existing information that we 
                do not want to lose.
            - We can again formalize the equation: `Q(s,a) <- Q(s,a)+a((r + future reward estimate) - Q(s,a))`
                - To estimate our future reward we wold make another call to our Q function: maxa'(s',a'):
                    - `Q(s,a) <- Q(s,a)+a((r + maxa'(s',a')) - Q(s,a))`
        - There are other additions we can make to the algorithm as well, to value reward now more than future rewards.
            - So we can add another parameter that discounts future rewards.
- The big picture idea of this entire formula is to say that every time we experience a new reward, we take that into
account by updating our estimate of how good the action is. In the future, we can make decisions based on the algorithm
    - once we have an estimate for every state, action and value of taking that action, then we can implement a `greedy
    decision-making` algorithm.
- **greedy decision-making**
    - when in state `s`, choose action `a` with highest `Q(s,a)`
    - there is a downside to this approach where we are able to find the solution, but it may not be the best or fastest
    method. if the AI always takes the action it thinks is the best, it will not know that another action may have a
    better outcome since it has never tried that action
- In reinforcement learning, there is a tension between **Exploration vs. exploitation**
    - **exploitation** refers to using knowledge that the AI already has.
    - **exploration** entails exploring other options that may not have been considered before.
    - therefore, an agent that only exploits may be able to get a reward, but it may not be able to maximize its rewards 
- One possible solution is known as the **ε-greedy algorithm**:
    - set `ε` equal to how often we want to move randomly (exploration)
    - with probability `1-ε`, choose the estimated best move.
    - with probability ε, choose a random move.
    - This can be quite powerful in a reinforcement learning context by not always just choosing the best possible move,
    but sometimes, especially early on, allowing random moves to explore.
    - over time, decrease the value of ε so the algorithm more and more often chooses the best move once it is more
    confident that it has explored all of the possibilities. 
- Common application of reinforcement learning: Game playing
    - reward signal at the end of the game. Win = 1, Lose = -1
    - e.g. Game of Nim.
        - every state is just 4 numbers, how many objects are in each of the 4 pile.
        - actions are the number of tiles removed from each individual pile.
        - reward happens at the end, win or lose.
    - In more complex games with more states and actions (e.g Chess), reinforcement learning will be much more difficult
    - often times in these cases, while we cannot learn the exact values for every state and action, we can approximate
    it. 
        - similar to minimax where we stop approximating at certain depths
    - **function approximation** - approximating `Q(s,a)`, often by a function combining various features, rather than
     storing one value for every state action pair.
        - allows the AI to approximate values by comparing two states, and if they look similar, then assume that
        actions that work in one of those states can also work in the other.
### Unsupervised learning
- **Unsupervised learning** - given input data without any additional feedback, learn patterns
- One task of unsupervised learning is **clustering**, or organizing a set of objects into groups in such a way that 
similar objects tend to be in the same group.
- Some clustering applications:
    - genetic research
    - image segmentation
    - market research
    - medical imaging
    - social network analysis
- One technique for clustering is called **K-means clustering**
- **K-means clustering** - algorithm for clustering data based on repeatedly assigning points to clusters and updating 
those clusters' centers.
    - define a cluster by its center, and assign points to that cluster by the closest cluster to that point.
    - need to define how many clusters we want for a data set
    - note that k-means clustering is an iterative process, and the cluster centers will be re-centered. Will be moved
    to the middle or average of the points assigned to that cluster (weighted by the total # of points).
    - once the cluster centers move, reassign the clusters that some of the points are assigned to.
    - repeat until the cluster centers do not move any more. We have reached an equilibrium where the algorithm can stop
         
    
    
    
    


                 
                        
                        
                
            
            
        