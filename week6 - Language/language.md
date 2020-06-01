## Language
- thus far we have interacted with AI by formulating problems in way that the AI can understand - learning to 'speak' 
the language of AI.
    - e.g. take a problem and formulate it as a search problem, or constraint satisfaction problem
- language is very challenging for AI and it encompasses a number of different types of tasks all under the broad
heading of `natural language processing`.
-  **natural language processing** - the idea of developing algorithms that allows our AI to be able to process
and understand natural language.
    - automatic summarization
        - auto-tldr bot.
    - information extraction
        - AI to extract meaningful semantic info out of content.
    - language identification
        - e.g. web browser automatic language ID and request to translate to english.
    - machine translation
        - language translation. e.g. google translate
    - named entity recognition
        - given a sequence of text can the AI pick out named entities (companies or people)
    - speech recognition
        - process audio and figure out words spoken there. Smart home devices: e.g. Siri, Google and Alexa
        - recall Hidden Markov model 
    - text classification
    - word sense disambiguation 
        - unlike in the language of numbers, where AIs have a precise representations of everything, words can be 
        ambiguous. Need to disambiguate and differentiate between different possible meanings of words.
### Syntax and Semantics
- **syntax** - structure of language and how that structure works.
    - comes easily to native speakers of that language.
    - doesn't just apply to natural languages, but programming languages as well.
    - determines if a sentence is _gramatically_ well-formed or not.
    - we might come up with rules pr ways to statistically learn these ideas.
    - syntax can also be ambiguous - the sentence can be both well-formed and not well-formed, there are certain ways 
    that we can take sentence and potentially construct multiple different structures for that sentence.
        - "I saw the man on the mountain with a telescope."
- **semantics** - idea of what does a word, sentence or essay mean?
### Context-Free Grammar
- **formal grammar** - a system of rules for generating sentences in a language.
    - not in terms of what they mean, but the syntactic structure. 
    - What structure of English are valid, correct sentences?
    - Although we as humans intuitively know what these rules are, it is going to be helpful to try and formally define 
    what the structures mean as well.
    - There are a number of different types of formal grammar across the Chomsky hierarchy of grammars.
- **context-free grammar** - a way of generating sentences in a language or via _rewriting rules_ - replacing one symbol
with other symbols.
- imagine a simple sentence: "she saw the city"
    - We know this is a valid, syntactically well-formed English sentence. But how do we encode this for the AI?
    - in order to answer a question, if we ask the AI 'what did she see?', we want the AI to recognize that she saw 
    the city.
    - each of the words will be called `terminal symbols`. This is what we care about generating.
    - each word will also be associated with a `non-terminal symbol`
        - not actually words in the language, but we use the **non-terminal symbols** to generate the terminal symbols
        - N - she
        - V - saw
        - D - the
        - N - city
- we use re-writing rules to translate **non-terminal symbols** to **terminal symbols**, for example the following rules
in our **context-free grammar**:
    - N -> she | city | car | Harry | ...
    - D -> the | a | an | ...
    - V -> saw | ate | walked | ...
    - P -> to | on | over | ...
    - ADJ -> blue | busy | old | ... 
- Other rules can be constructed that are not just one non-terminal translating into a terminal symbol. We will always
have one non-terminal symbol on the left side of the arrow, but on the right side we can have other non-terminal 
symbols in addition to the terminal symbols we have seen.
    - **noun-phrase** - e.g. the city, the car on the street, the big city
        - NP -> N | D N
            - e.g. NP - N - she
            - e.g.    NP
                   ___|___
                   |     |
                   D     N
                   |     | 
                  the   city
    - **verb-phrase** - e.g. I walked, saw the city
        - VP -> V | V NP
            - e.g. VP - V - walk
            - e.g.     VP
                   ____|_______
                   |          |
                   V         NP
                   |       ___|___ 
                  saw      |     | 
                           D     N
                           |     |
                          the   city
- basic structure of a a sentence
    - S -> NP VP
- by defining a set of rules, we are able to run algorithms now that can take the words (**CYK algorithm**) where the 
computer is able to start with a set of terminal symbols (e.g. she saw the city) and then using the rules to figure out
how to transition from a sentence to those terminal symbols.
- this set of rules are good for dealing with simple sentences, but we can imagine there are more complex scenarios 
that will need to be handled with an expanded set of rules.
### nltk
- python library short for **natural language toolkit**.
- has the ability to parse through a **context-free grammar** to take words and construct a `syntax tree`
- Example of some additional complexity:
    - AP -> A | A AP
    - NP -> N | D NP | AP NP | N PP
    - PP -> P NP
    - VP -> V | V NP | V NP PP
- ambiguous sentences: "She saw a doc with binoculars"
### n-grams
- **_n_-gram** - a contiguous sequence of _n_ items from a sample of text.
    - those items may take on various different forms
- **character _n_-gram** - a contiguous sequence of _n_ characters from a sample of text.
    - e.g. 3 characters in a row
- **word _n_-gram** - a contiguous sequence of _n_ words from a sample of text.
- when looking at a single word or character: **unigram**
- **unigram** - a contiguous sequence of 1 item from a sample of text.
- **bigram** - a contiguous sequence of 2 items from a sample of text.
- **trigram** - a contiguous sequence of 3 items from a sample of text.
- e.g. "How often have I said to you that whaen you have eliminated the impossible whatever remains, however improbable,
 must be the truth"
    - trigrams: "How often have", "often have I", "have I said", etc.
- When extracting text, it is not meaningful to extract the entire text at one time, but rather segment the text into 
pieces that we can begin to do analysis on.
### Tokenization
- we need a way to extract the n-grams, given an input we need to separate into the individual words:
- **tokenization** - the task of splitting a sequence of characters into pieces (tokens).
    - most commonly refers to word tokenization, but may also come up in the context of **sentence tokenization**
- **word tokenization** - the task of splitting a sequence of characters into words.
    - e.g. given an input sequence, use the python split() method.
    - naive approach: split a sentence up for every 'space' character.
    - "Whatever  remains, however improbable, must be the truth."
        - result = ["Whatever", "remains,", "however", "improbable,", "must", "be", "the", "truth."]
        - notice that we keep the punctuations. Poses a challenge when we compare words to each other "truth" =/= 
        "truth."
            - treat punctuations as separate tokens all together, or strip them as well.
            - but it gets trickier with words that have apostrophes "o'clock" or hyphenated words.
- similar issues with **sentence tokenization**:
    - split sentences with periods, explanation points or question marks, which are types of punctuation that we know
    come at the end of sentences.
    - gets trickier with words like "Mr." or "Mrs.", or even with dialogue.
- in practice, there are heuristics that we can use. We know that there are certain occurrences of periods, e.g. the 
one after Mr. or Mrs. that we know are not beginnings of new sentences.
- once we can tokenize passages, we can then start to extract what the n-grams actually are using the **nltk** library.
### Markov Models
- once we have the n-grams, data about the frequency of sequences of words show up in a particular order.
- using the data we can start doing predictions:
    - if we see 'it was', it is reasonable to predict that the next word is 'a', etc. 
        - since we have the data on tri-grams, and based on two words, we can start to make predictions on the third 
        word.
- recall that **markov models** refer to a sequence of events that happen one time step after one time step.
    - e.g. every unit has some ability to predict what the next unit is going to be, or the past two units predict 
    what the next unit is going to be.
- can use a markov model and apply it to language for a naive and simple approach at generating natural language and 
getting our AI to be able to speak English-like text.
- ask the AI to come up with a probability distribution. Given two words, what is the probability distribution over 
what the third word could be based on all of our data?
    - the effect of the Markov model is to begin generate text that is not in the original corpus, but sounds like the 
    original corpus. It is using the same rules that the original corpus was using.
- can use a library called **Markovify**
- now we can start to look at other tasks that we want our AI to perform: **text categorization**
    - can also be thought of as a classification problem.
### Bag-of-Words Model
- Given an email and we want to decide whether or not it belongs in the inbox or in spam?
    - look at the text and perform an analysis on that text to draw some conclusions.
- **sentiment analysis** - where we analyze a sample of text, does it have a positive or negative sentiment?
    - e.g. product reviews on a website, feedback on a website
    - ultimately depends on the individual words in each review. 
    - Look at keywords that are more likely positive or negative. ignore the structure of the sentence 
    - (e.g. how the words relate to each other or parse the sentences to construct grammatical structure).
- **bag-of-words model** - model that represents text as an unordered collection of words.
    - ultimately, we do not care about the order or structure of the words. We dont care about what noun goes with 
    what adjectives or how things agree with each other.
    - works very well for sentiment analysis.
### Naive Bayes
- one approach to try to analyze the probability that something is positive or negative sentiment.
- can use this to categorize text into possible categories. 
- based on Bayes' Rule:
    - `P(b|a) = P(a|b)*P(b) / P(a)`
    - definition of conditional independence and looking at what it means for two events to happen together.
- P(Positive) or P(Negative):
    - P(Positive|"my grandson loved it") => P(Positive|"my", "grandson", "loved", "it") 
        - according to the **bag-of-words model** ignore the ordering/sequence of words
    - according to Bayes' rule:
        - P(Positive|"my", "grandson", "loved", "it") _equal to_ 
            - (P("my", "grandson", "loved", "it"|Positive)*P(Positive)) / P("my", "grandson", "loved", "it")
        - because the denominator does not change, we can say that the two expressions are proportional:
            - P(Positive|"my", "grandson", "loved", "it") _proportional to_ 
             P("my", "grandson", "loved", "it"|Positive)*P(Positive)
        - using the denominator would get us an exact probability. We can figure out what the probability is 
        proportional to, and at the end, we normalize the probability distribution - ensure it sums up to 1.
    - recall that we can calculate this as a **joint probability** of all of these things happening:
        - P(Positive, "my", "grandson", "loved", "it")
    - ultimately, we can say that P(Positive|"my", "grandson", "loved", "it") _proportional to_ 
     P(Positive, "my", "grandson", "loved", "it")
 - how do we calculate this **joint probability**? Recall that we can multiply all of the _conditional probabilities_, 
 but ultimately, this is a very complex calculation and difficult to solve.
 - Simplify the notion, rather than calculate an exact probability distribution, assume the words are independent of 
 each other if we already know that it is a positive message. 
    - if it is a positive message, it does not change that the word "grandson" is in our message, if i know "loved" is
    in the message.
    - may not be 100% true in practice, but is a good simplification that leads to pretty good results.
    - ASSUME: the probability of all of these words showing up depend **only** on whether it's a positive or negative.
        - e.g. the probability that "loved" shows up will not change given the fact that "my" is in the message as well.
    - Assume the original probability is _naively proportion to_ 
    P(Positive)*P("my"|Positive)*P("grandson"|Positive)*P("loved"|Positive)*P("it"|Positive)
        - these are numbers that we can ultimately calculate given some data.
- Given a data set with labeled reviews, we can then calculate the terms above:
    - P(Positive) = num positive / num total
    - P("loved"|Positive) = num positive with "loved" / num positive
    - e.g. Calculation for: P(Positive)*P("my"|Positive)*P("grandson"|Positive)*P("loved"|Positive)*P("it"|Positive)
        - Positive = 0.49; Negative = 0.51
        - |        |Positive|Negative|
          |--------|--------|--------|
          |   my   |  0.30  |  0.20  |
          |grandson|  0.01  |  0.02  |
          |  loved |  0.32  |  0.08  |
          |   it   |  0.30  |  0.40  |
        - Positive = 0.49 * 0.30 * 0.01 * 0.32 * 0.30 = 0.00014112
            - by itself, not a meaningful number, but meaningful when compared to negative sentiment messages
        - Negative = 0.51 * 0.20 * 0.02 * 0.08 * 0.40 = 0.00006528
        - Treat these two values as a probability distribution and normalize them.
            - Probabilities: Positive = 0.6837; Negative = 0.3163
- After applying naive Bayes', we end not with a categorization, but a _confidence level_
- drawback: applying this rule as is for data sets that contain 0.00 (e.g. no positive messages ever had the word).
    - will lead to 0.00 result for that Positive or Negative probability.
    - number of possible ways to ensure that we never multiply by 0.
    - **additive smoothing** - adding a value _a_ to each value in our distribution to smooth the data.
    - **Laplace smoothing** - adding 1 to each value in our distribution: pretending we've seen each value one more 
    time than we actually have.
### Information Retrieval
- **information retrieval** - the task of finding relevant documents in response to a user query.
- e.g. search engine, library catalog. Searching for documents that match our query.
    - first, we would need to take documents and figure out what are the documents about.
- **topic modeling** - models for discovering the topics for a set of documents.
- it may be intuitive to use **term frequency** for this task:
    - **term frequency** - number of times a term appears in a document.
        - e.g. if a document has 100 words and a term appears 10 times, it can be said to have a term frequency of 10.
        - can also be seen as a proportion, where you have a term frequency of 0.1 or 10%.
- we can use the **tf-idf** library to determine term frequencies in a corpus of documents.
### tf-idf
- different categories of words:
    - **function words** - words that have little meaning on their own, but are used to grammatically connect other 
    words.
        - e.g. am, by, do, is, which, with, yet, ...
        - fixed list that does not change very often.
    - **content words** - words that carry meaning independently.
        - e.g. algorithm, category, computer, ...
- strategy to ignore the **function words**
- we ultimately want to know the words that show up frequently in this document, that show up less frequently in other
documents to be able to get an idea of the topics discussed within the document.
    - **inverse document frequency** - measure of how common or rare a word is across documents.
        - mathematically it is usually calculated as: `log(totalDocuments / numDocumentsContaining(word))`
            - e.g. if the total number of docs and number of docs containing a word. 
            - If all documents contain a word (e.g. holmes), taking the logarithm of 1 is 0.
- **tf-idf** - ranking of what words are important in a document by multiplying term frequency (TF) by inverse document
 frequency (IDF).
    - the importance of a word depends on two things:
        - how often it shows up in the document (TF)
        - rarity of the word in a document compared to the lot of documents (IDF)
### Information Extraction
- with **tf-idf**, we are starting to jump into the world of semantics, what it is that things actually mean, how the 
words relate to each other and in particular, how we can extract info out of the text.
- **information extraction** - the task of extracting knowledge from documents.
- Provide the AI templates to allow it to search a corpus of a document that match the template:
    - e.g. When {company} was founded in {year},
        - requires us to figure out what is the structure of the info we are looking for. Which may be difficult to know
        - different websites and authors will do this differently. this method is not going to be able to extract
        all of the information. 
            - if words are in a slightly different order it will no longer match our template.
    - instead of giving the template, we can provide the AI the data:
        - e.g. tell the AI, Facebook was founded in 2004 and Amazon was founded in 1994 and then set the AI loose to
        search through the documents. The AI can then discover the templates themselves.
- giving the AI enough data, it can create and discover templates to extract information to generate knowledge. The 
more knowledge it learns, the more new templates it's able to construct as it looks for constructions that show up
elsewhere as well.
- information extraction is another powerful tool, but it only works in limited contexts when the AI is able to find 
information that match the format of the template it expects and is able to connect to that pair of data.
- But how do we move past this and come up with a definition for all works, to be able to relate all of the words in a 
dictionary to each other. Ultimately, this is necessary to for our AI to be able to communicate - need a representation 
of what words actually mean.
### WordNet
- researchers curated together a list of words, their definitions and their various different senses (if they 
have multiple meanings) and also how the words relate to one another.
- comes built into the **nltk** library.
    - e.g. the word 'city' has three `senses` or meanings, as well as categories that the word belongs to. 
        - The categories ultimately allows us to relate words to other words.
- although helpful, this does not scale particularly well as language changes or all the various different 
relationships that words may have with one another.
### Word Representation
- we want a way to represent a meaning of a word in a way that our AI is going to be able to do something useful with.
- anytime we want the AI to be able to look at text, we want the AI to understand that text, to relate text and
 similar words and be able to understand the relationship between those words. 
- We want a way for the computer to **represent** the information.
    - whenever we want the AI to represent something, it can be helpful to represent it using numbers
        - e.g. winning and losing in a game -> -1, 0, 1
        - e.g. take data and turn it into a vector of features
- if we want to pass the words into a neural network (e.g. for translation or classification), we need to represent
words as vectors or numbers.
- **one-hot representation** - representation of meaning as a vector with a single 1, and with other values as 0.
    - the location of the 1 tells the AI the meaning of the word.
    - e.g. "He wrote a book."
        - he [1, 0, 0, 0]
        - wrote [0, 1, 0, 0]
        - a [0, 0, 1, 0]
        - book [0, 0, 0, 1] 
    - limitations: vectors can get very very big - not a tractable way of representing numbers. Ideally we want the 
    vectors to represent meaning in a way that allows information/meaning to be extracted.
        - e.g. "he wrote a book." and "he authored a novel" will have different vectors, but ultimately have the same
        meaning. 
            - goal: to have "wrote" and "authored", "book" and "novel" to have the same vectors (b/c similar meanings)
- **distributed representation** - representation of meaning distributed across multiple values.
    - e.g. "He wrote a book."
        - he [-0.34, -0.08, 0.02, -0.18, 0.22, ...]
        - wrote [ -0.27, 0.40, 0.00, -0.65, -0.15, ...]
        - a [-0.12, -0.25, 0.29, -0.09, 0.40, ...]
        - book [-0.23, -0.16 , 0.05, -0.57, ...]
    - now we can hope that "wrote" or "authored" have vectors that are pretty close to one another. the "distances" are
    not too far apart.
    - the goal of a lot of statistical machine learning and approaches to  NLP is about using the vector representation 
    of words.
    - we can define a word in terms of the words that show up around it. we can get the meaning of a word by the 
    context in which that word happens to appear.
        - e.g. for ___ he ate. can fill in with breakfast, lunch dinner.
        - if two words show up in a similar context, then those two words are probably related to each other.
### word2vec
- **word2vec** - model for generating word vectors.
    - give `word2vec` a corpus of documents and it will produce vectors for each word.
        - does this with `skip-gram architecture`
    - **skip-gram architecture** - neural network architecture for predicting context words given a target word.
        - using a large neural network - we have one input cell (node) for every word.
        - goal: given target word, predict a context word.
            - will do so using methods we've seen before. back-propagating weights through the neural network.
            - if we use a single layer of hidden nodes, each hidden node will represent the vector of numerical 
            representation for that word.
            - general idea: if two words are similar that show up in similar contexts (use the same target words) will
            have similar vector values for the hidden nodes
- we measure the effectiveness of the algorithm by looking at the **distance** between different words.
    - one definition of distance can be the `cosine distance`, measuring the angle between vectors (0 = close, 1 = far).
    - this ides is incredibly powerful in representing the meaning words in terms of their relationship to other words.
- because these are not vectors, we can calculate the relationships between various different words.
    - e.g. 'king' and 'man'. we can subtract 'king' - 'man' and get a new sequence of vectors. 
        - This new sequence of num will tell us what we need to do to get from man to king.
        - we can then take this value and add it to another value, e.g. 'women'. We can ask what value we expect to get
        when we perform this calculation: `women + (king - man)` = **queen**
- the idea of representing words as vectors is incredibly useful and powerful anytime we want to do statistical work with
natural language.