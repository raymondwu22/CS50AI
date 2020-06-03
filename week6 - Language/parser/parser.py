import nltk
import sys
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP CP

VP -> V | V NP | VP NP | VP PP | VP Adv | Adv VP | Adj VP
NP -> N | Adv N |  Adv NP | Adj N | Adj NP | Det N | Det NP | N Adj | N Adv | N PP | N P S
PP -> P NP
CP -> Conj S | Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def contains_letters(phrase):
    return bool(re.search('[a-zA-Z]', phrase))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # raise NotImplementedError
    sentence = sentence.lower()
    result = [word for word in nltk.word_tokenize(sentence) if contains_letters(word)]
    return result


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # raise NotImplementedError
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            chunk = subtree.subtrees()
            # start at -1 to account for NP label of the subtree
            count = -1
            # loop through the subtrees subtree
            for item in chunk:
                if item.label() == 'NP':
                    count += 1
            if count == 0:
                chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
