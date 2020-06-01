## Uncertainty
- Week 1 looked into how AI represents `knowledge` in the form of logical sentences
    - want our AI to represent knowledge/information and use those pieces of information to derive new pieces of 
    information via inferences
    - to be able to take some information and deduce some conclusions based on what it knew for sure
- but very rarely are our computers going to know things for sure. may believe things with some probability, but not 
100% certain
- we want to use the information the computer has some knowledge about, even if not perfect, to be able to make some 
inferences and draw conclusions
    - e.g. a mars rover wont know exactly where it is, or what is around it. But there is data to allow it to draw 
    inferences with some probability
    - e.g. predicting the weather. Will not know what tomorrow's weather will be with 100% certainty, but can infer with
     some probability what tomorrow's weather will be from today or yesterday's weather. 
- Games, element of chance in the games. e.g. rolling a dice

## Probability theory
- mathematical foundations and key concepts
- how can we use probability and the mathematical ideas to represent ideas in models that we can put into an AI to use
 probability to make inferences
- Possible worlds: `ω`
    - e.g. rolling a die we have 6 `possible worlds`: 1 - 6
        - each possible world has a probability of being true: `P(ω)`
- Basic axioms of probability:
    1. 0 ≤ P(ω) ≤ 1
    2. Σ P(ω) = 1
    - e.g. P(2) = 1/6
- Probability is simple with the example of one die, but gets trickier when the models of the world get more complex.
    - e.g. 2 die; care about the sum of the two rolls
    - P (sum to 12) = 1/36
    - P (sum to 7) = 6/36 = 1/6
    - representing events that are more or less likely. Judgements where we figure out in the abstract the probability
     that something takes place are known as `unconditional probabilities`
- `unconditional probability` - degree of belief in a proposition in the absence of any other evidence
    - e.g. without knowing any other information. e.g. if we roll a die, whats the chance it will be a 2? 
- usually when we're thinking about probability, especially if we are training an AI to intelligently know something 
about the world and make predictions based on that information, it is `conditional probability`
    - `conditional probability` - degree of belief in a proposition given some evidence that has already been revealed
- `conditional probability`
    - P(a|b)
        - we want the probably that a is `true`
        - the right side of the bar is the evidence. `b` is true.
        - read as: `what is the probability of a, given b?`
    - e.g. P(rain today|rain yesterday)
    - P(route change|traffic conditions)
    - P(disease|test results)
    - notion of conditional probability comes up very often. We would like to reason about some question, but being 
    able to reason a little more intelligently, by taking into account evidence that we already have.
    - how do we calculate conditional probability?
        - `P(a|b) = P(a∧b)/P(b)`
            - if i want to know the probability that a is true given b is true, I want to consider all the ways both 
            are true, out of the only worlds I care about, where b is already true.
                - ignore all cases where b isnt true because it isnt relevant to my ultimate computation.
            e.g. rolling 2 dices and P(sum to 12)
                - remember that is 1/36
                - P(sum 12|one die is 6) = ?
                    - need to know the probability that both variables are 6 = 1/36
                    - to ge the conditional probability: (1/36) / (1/6) = 1/6
        - also expressed as: `P(a∧b) = P(b) * P(a|b)` or `P(a∧b) = P(a) * P(b|a)`
- sometimes when we deal with probability, we don't just care about the `Boolean`
 event, but rather the variable values in a probability space
    - `random variable` - a variable in probability theory with a domain of possible values it can take on
        - Examples:
            - roll: {1,2,3,4,5,6} and the probability of each outcome is the same
            - weather: {sun, cloud, rain, wind, snow} and each of these have a different probability
            - traffic: {none, light, heavy}
            - flight: {on time, delayed, cancelled}
        - `Probability distributions` (see above), takes a random variable, and gives the probability for for each 
        possible value in the domain 
            - flights:
                -  P(Flight = on time) = 0.6
                -  P(Flight = delayed) = 0.3
                -  P(Flight = cancelled) = 0.1
            - concise example with a vector (sequence of values)
                - *B*(Flight)= <0.6,0.3,0.1>
                    - Note**: we would need to know the order of this vector to interpret the values
- `independence` - the knowledge that one event occurs does nto affect the probability of the other event
    - e.g. in the context of a two dice roll, the two rolls are independent of each other. Knowing the result of one die
    does not provide any additional information about what the value of the second die.
    - not always the case, e.g. in the case of weather, if it is cloudy, it may increase the probability of raining.
        - some information informs some other event or some other random variable
    - mathematically independence can be defined as: `P(a∧b) =  P(a)*P(b)` 
        -e.g. P(6∧6) = P(6)P(6) = 1/6 * 1/6 = 1/36

### Bayes' Rule
- Let's go back to the equations we saw earlier:
    - `P(a∧b) = P(b) * P(a|b)` 
    - `P(a∧b) = P(a) * P(b|a)`
    - We can set both equations equal to each other: 
        - P(a) * P(b|a) = P(b) * P(a|b)
        - `Bayes Rule`: P(b|a) = (P(b)*P(a|b)) / P(a)
            - **Very important** when it comes to trying to infer things about the world because it means you can express 
            one conditional probability, the conditional probability of b given a, using knowledge about the probability 
            of a given b. We are using the reverse of that conditional probability
- Example:
    - Cloudy in the morning                     Raining in the afternoon
    - **Question**: Given clouds in the morning, what's the probability of rain in the afternoon?    
        - Additional information: 
            - 80% of rainy afternoons start with cloudy mornings.
            - 40% of days have cloudy mornings.
            - 10% of days have rainy afternoons.
        - P(rain|clouds) = (P(clouds|rain)*P(rain)) / P(clouds)
            = (0.8)(0.1)/0.4 = 0.2 = 20%
- often useful when one of the conditional probabilities might be easier for us to know about or easier to have data 
about and using that information we can calculate the other conditional probability
    - Knowing `P(cloudy morning|rainy afternoon`), we can calculate `P(rainy afternoon|cloudy morning)`
    - More generally, if we know the probability of some visible effect, given some unknown cause we are unsure about,
     then we can calculate the probability of that unknown cause, given that visible effect
        - Knowing `P(visible effect|unknown cause`), we can calculate `P(unknown cause|visible effect)` 
        - Example: Knowing `P(medical test result|disease`), we can calculate `P(disease|medical test result)`
        - Example: Knowing `P(blurry text|counterfeit bill`), we can calculate `P(counterfeit bill|blurry text)`
- So far, we've discussed different types of probability
    - unconditional: what is the prob of this event occurring
    - conditional: we have some evidence and we would like to using the evidence, be able to calculate the probability 
    as well
- `joint probability`: likelihood of multiple different events simultaneous
    - Example: probability distributions:
                     AM                         PM
            | C=cloud | C=¬cloud |      | R=rain | R=¬rain |
            |---------|----------|      |--------|---------|
            |   0.4   |    0.6   |      |   0.1  |   0.9   |
        - using just these pieces of info, we do not know how these relate to each other. Unless we have joint probability;
        for every combination of the two tables. 
                         AM&PM
            |         |  R=rain  |  R=¬rain  |
            |---------|----------|-----------|
            | C=cloud |   0.08   |    0.32   |
            | C=¬cloud|   0.02   |    0.58   |
        - using the joint probability table, we can begin to draw other pieces of info like conditional probability
            - e.g. P(C|rain)
            - remember, we can calculate this with:
                P(C|rain) = P(C, rain) / P(rain) = `aP(C,rain)`
                - dividing by the probability of rain, is dividing by a constant. Often times we can not worry about 
                what the exact value is, and just know that it is a constant value. 
                    - we can simply express the formula as: `a*P(C, rain)`
                - the conditional distribution P(C|rain) is proprotional to the joint probability P(C,rain)
                    - `a<0.08, 0.02>`
                        - of course 0.08 and 0.02 do not sum up to 1, we know we need a constant `a` to normalize our 
                        data, 10 in this case: a<0.08,0.02> = 10<0.08,0.02> = `<0.8,0.2>`
### Probability Rules
1.**Negation**: `P(¬a) = 1-P(a)`
2. **Inclusion-Exclusion**: `P(avb) = P(a)+P(b)-P(a∧b)`
    - think of a Venn diagram where we need to remove the overlapping middle that double-counts the situation.
3. **Marginalization**: `P(a) = P(a,b)+P(a,¬b)`
    - can figure out the probability of a from 2 disjointed cases with regards to b
    - ultimately does not matter what b is, or how it relates to a, so long as we know these joint 
    distributions
    - Sometimes, it may not be a Boolean event, but rather a broader probability distribution with multiple values
        - we would need to sum up all different cases in this case:
            - `P(X = xi) = Σ P(X=xi, Y=yj)`
            - Example:
                |         |  R=rain  |  R=¬rain  |
                |---------|----------|-----------|
                | C=cloud |   0.08   |    0.32   |
                | C=¬cloud|   0.02   |    0.58   |
                P(C=cloud)
                = P(C=cloud,R=rain)+P(C=cloud,R=¬rain)
                = 0.08 + 0.32 
                = 0.40
    - marginalization gives us the ability to go from joint distributions to individual probabilities that we may
    care about
4. **Conditioning** - `P(a) =P(a|b)*P(b) + P(a|¬b)*P(¬b)`
    - we do not have their joint probabilities, but we have access to their conditional probabilities 
    - Just like in the case of marginalization, where there was an equivalent rule for random variables that could take 
    on multiple possible values in a domain of possible values
        - `P(X = xi) = Σ P(X=xi | Y=yj) * P(Y=yj)`

### Bayesian Networks
- An example probabilistic model
- `Bayesian network` - data structure that represents the dependencies among random variables
    - odds are most random variables are not independent of each other, that there's some relationship between
    things that are happening that we care about.
        - e.g. if it is raining today, that might increase the likelihood that my flight or my train gets delayed. There
        is an inherent dependency between these random variables
        - a `Bayseian network` is able to capture these dependencies
    - characteristics:
        - directed graph
        - each node represents a random variable
        - arrow from X to Y means X is a parent of Y
        - each node X has a probability distribution `P(X|Parents(X))`
            - think of the parents as a causes for some effect that we will observe
    - Example: Have an appointment out of town, and I need to take a tran to get to that appointment. So
    what are the things I care about? Getting to the appt on time to attend it, or miss the appt. Obviously this 
    variable is influenced by the train. Either the train is on time, or delayed. But the train itself is influenced,
    whether it's on time or not depends on the rain, or other variables too, e.g. maintenance on the train tracks.
    - 4 nodes, each of which has a domain of possible values:
                        Rain
                {none, light, heavy}
                    |           |
                    v           |
               Maintenance      |
                    |           |
                    v           v
                        Train
                 {on time, delayed}
                          |
                          v
                     Appointment
                    {attend, miss}
        - The arrows, the edges pointing from one node to another to encode dependence inside this graph.
            - e.g. Making it to the appointment is dependent upon whether the train is on time or delayed.
            Whether the train is on time or delayed is dependent on whether or not where was maintenance on the train 
            track and if it was raining. Whether or not there was maintenance is dependent on rain itself.
            - Idea is that we can come up with a probability distribution for any of these nodes only based upon its 
            parents
        - starting with the Rain node. Note there are no arrows pointing into it meaning its probability distribution
         is not going to be a conditional distribution over the possible values for the Rain random variable.
        -               Rain            |  none   |  light   |   heavy   |
                {none, light, heavy}    |---------|----------|-----------|
                         |        |     |   0.7   |    0.2   |    0.1    |
                         |        |       
                         V        |
        -           Maintenance   |     |    R    |   yes   |    no    |
                     {yes, no}    |     |---------|---------|----------|
                         |        |     |  none   |   0.4   |   0.6    |
                         |        |     |  light  |   0.2   |   0.8    |
                         |        |     |  heavy  |   0.1   |   0.9    |
                         v        v       
                             Train              |    R    |    M    | on time | delayed |
                        {on time, delayed}      |---------|---------|---------|---------|
                               |                |  none   |   yes   |   0.8   |   0.2   |   
                               |                |  none   |    no   |   0.9   |   0.1   |
                               |                |  light  |   yes   |   0.6   |   0.4   |
                               |                |  light  |    no   |   0.7   |   0.3   |
                               |                |  heavy  |   yes   |   0.4   |   0.6   |
                               |                |  heavy  |    no   |   0.5   |   0.5   |
                               v
                          Appointment           |    T    | attend  |   miss  |
                         {attend, miss}         |---------|---------|---------|
                                                | on time |   0.9   |   0.1   |
                                                | delayed |   0.6   |   0.4   |
                                                
        - After describing the structure of the Bayesian Network and relationship between each of the nodes by 
        associating each of the nodes in the network with a probability distribution, whether it's an unconditional
        probability (e.g. root node: Rain) or a conditional probability (all others), dependent on the values of the
        parents. We can begin to do some computation and calculation using information inside of that table.
        - Computing Joint Probabilities
            - P(light)
                - rain is a root node, therefore we can just look at the probability distribution for rain and extract 
                from the node, the probability of light rain.
            - P(light, no)
                - start with probability of light rain, and also want probability of track maintenance
                    - what we really want to know is the probability of no track maintenance given that there is light
                    rain
                - P(light) P(no|light)
                    - take unconditional probability of light rain and multiply it by the conditional probability of 
                    no track maintenance given light rain.
            - P(light, no, delayed)
                - P(light) P(no|light) P(delayed|light,no)
            - P(light, no, delayed, miss)
                - P(light) P(no|light) P(delayed|light,no) P(miss|delayed)

### Inference
- what we really want to do is get new pieces of information (INFERENCE)
    - in the context of knowledge, we considered the problem of inference. Given things that I know to be true, 
    can I draw conclusions, make deductions about other facts about the world that I also know to be true
    - we will apply the same ideas to probability. Using information that I do know, whether some evidence 
    or probabilities, can I figure out the probabilities of other variables taking on particular values?
- Query *X*: variable for which to compute distribution
    - e.g. probability that I miss my train or probability that there is track maintenance
- Evidence variable *E*: observed variables for event `e`
    - e.g. observed that there is light rain, and using that evidence what is the probability that my train is 
    delayed?
- Hidden variables *Y*:
    - e.g. I know if its raining, and I want to know whether my train is delayed or not? The hidden variable will be 
    if there is maintenance on the track or if I will make it to my appointment
- Goal: Calculate P(X|e)
    - we can do this calculation using a lot of the probability rules we've seen above. Ultimately we are going to look 
    at the math at a high level
- Example: P(Appointment|light,no)
    - Query: Appointment
    - Evidence: light, no
    - Hidden: train
    - Remember that a conditional probability is proportional to the joint probability.
        - Recall: P(a|b) = `αP(a∧b)` (alpha is a constant)
    - `P(Appointment|light,no) = α * P(Appointment, light, no)`
        - α multiplied by joint probability (likelihood of multiple different events simultaneous)
        - use `marginalization trick` here since we are missing the Train variable. 
            - sum up both possibilities: if the train is on time or if it is delayed.
            - iterate over all possible values for the hidden variable Train
        - α [P(Appointment, light, no, on time) + P(Appointment, light, no, delayed)]
            - need to normalize the data at the end to ensure it adds up to 1.
- Formula for `inference by enumeration`:
    - P(X|e) = α*P(X,e) = α Σy P(X,e,y)
        - *X* is the query variable.
        - *e* is the evidence. 
        - *y* ranges over values of hidden variables. (*marginalize* over y)
        - *α* normalizes the result.
    - There are a lot of libraries that help with probabilistic inference and be able to take a Bayesian network
    and do the underlying math/calculations.
        - Example: `pomegranate` library
    - CON: not very efficient. Can optimize by avoiding repeated work.
        - But as the number of variables and number of possible values gets larger, there will be a lot of computations 
        to do the inference.
        - For this reason, we do not always care about the exact inference, or the exact probability, but rather if we 
        are able to approximate the inference procedure it can be pretty good as well.
- `Approximate Inference` through *Sampling*
    - Take a *sample* of all of the variables inside the Bayesian network. 
    - Sample one of the values from each of the nodes according to their probability distribution
        - start at the root and a random number generator to randomly pick one of the values
            - e.g. pick none
        - with the rest of the dependent nodes, we only sample the distribution from what was selected in the 
        parent/root node.
    - Sampling becomes powerful when we repeat this thousands of times and can get the probability for each of the
    possible variables that could come up.
    - From our sampling, we may want ask the probability for both unconditional and conditional probabilities:
        - *unconditional*: P(Train = on time)
        - *conditional*: P(Rain=light | Train=on time)
            - reject all samples where the train was delayed.
            - from the remaining cases, what is the probability of light rain given the train was on time?
            - also known as `rejection sampling`
                - CON: if the evidence is a very unlikely event, we will be rejecting a lot of samples (inefficient)
    - `Liklihood Weighting`
        - tries to address the inefficiencies of `rejection sampling`.
        - Start by fixing the values for evidence variables. (Do not sample these).
        - Sample the non-evidence variables using conditional probabilities in the Bayesian Network.
        - Weight each sample by its *likelihood*: the probability of all of the evidence.
        - Example: P(Rain=light|Train=on time)
            - Start by fixing the evidence that the Train is on time. 
            - ONLY sample those cases where the train is on time. Will not have to throw our any results since we are 
            only sampling those cases with our expected variable.
            - The weight is based on how probable the train is on time (the value of the evidence variable)
    - We've spoken about particular variables as it pertains to particular variables that have these *discrete values* 
    - But we have not considered how values may change over time.

### Uncertainty over time
- One quick example is the context of weather.  
    - Sunny days vs. rainy days. We want to know not just the probability of rain now, but also the forecast for the 
    next few days or weeks.
    - We will have a random variable, not just one for the weather, but every possible time step (e.g. days).
    - *Xt*: weather at time t
        - due to the large data set, it's helpful to make simplifying assumptions about the problem that we can 
        assume are true, to make our lives easier.
            - does not have to be completely accurate assumptions, but if they're close to accurate or approximate, 
            they're usually pretty good.
            - The assumption is known as the `Markov assumption`
- *Markov assumption* - the assumption that the current state depends on only a finite fixed number of previous states.
    - e.g. the current day's weather depends not on all the previous day's weather, but I can predict based on 
    yesterday's weather.
    - by putting a bunch of the random variables together using the Markov assumption, we can create a `Markov Chain`
- *Markov chain* - a sequence of random variables where the distribution of each variable follows the Markov assumption 
    - e.g. using today's weather, I can come up with a probability distribution for tomorrow's weather
    -                       tomorrow(X+1)
                |       | sunny | rainy | 
                |-------|-------|-------|
      today(X)  | sunny |  0.8  |  0.2  |
                | rainy |  0.3  |  0.7  |
        - if today is sunny, then tomorrow is likely to be sunny and vice versa.
        - We will be using this matrix as a `transition model` from one state to the next
            - The pattern this Markov chain follows is that when it's sunny, it tends to stay sunny for a while and vice 
            versa. 
            - we can do analysis on this as well.
                - Given today is raining, what is the probability that tomorrow is also raining?
                - Or ask probability questions: what is the probability of this sequence values (sun, sun, rain, rain,
                 rain)?
                - also many python libraries for this (e.g. pomegranate)
            - Markov models rely on us knowing the values of these individual states (e.g. today is sunny or raining)
                - But in practice, this often is not the case. We do not know for certain what the exact state of the 
                world is. However, we are able to "sense" some information of that state (e.g. camera or microphone).
                    - The "sensor" will provide data related to the state of the world even though it does not provide
                    the underlying true state of the world. 
- *Sensor Models* - how do we translate the hidden state (underlying true state) from the observation (what we/our AI 
has access to)
    - |  hidden state  |  observation   | 
      |----------------|----------------|
      |robot's position|   sensor data  |
      |  words spoken  | audio waveforms|
      |user engagement | analytics data |
      |    weather     |    umbrella    |
        - what the robot's true position is influences the observations (sensor data). 
        - similarly, with voice recognition software, devices will use audio waveforms to determine
        what was actually spoken (or what was likely spoken)

- To try and model this type of idea with hidden states, rather than utilizing a Markov Model has states connected by
a transition matrix, we will use a `hidden Markov model`
- *Hidden Markov Model* - a Markov model for a system with hidden states that generate some observed event.
    - In addition to that transition model, we also need another model that gives us an observation (e.g. did an
     employee) 
     - Using the observation we can begin to predict with reasonable likelihood, what the underlying state is. Even if 
     we do not get to observe the underlying state itself.
     - also known as a *Sensor Model* or *Emission probabilties*
 - *sensor Markov assumption* - the assumption that the evidence variable depends only on the corresponding state.
    - evidence variable = the thing we observe
    - this assumption might not hold in practice, but for simplification, it can be helpful to apply this assumption to 
    reasonable about the approximation.
    - You can try to infer something about the future or the past or about the hidden states that may exist
        - all of these are based on the same idea of conditional probabilities and using probabilities that we have to 
        draw conclusions.
        - |         task          |                               definition                                         | 
          |-----------------------|----------------------------------------------------------------------------------|
          |        filtering      |given observations from start until now, calculate distribution for current state |
          |        prediction     |given observations from start until now, calculate distribution for a future state|
          |        smoothing      |given observations from start until now, calculate distribution for a past state  |
          |most likely explanation|given observations from start until now, calculate most likely sequences of states|
        - `most likely explanation` is very commonly performed for voice recognition 