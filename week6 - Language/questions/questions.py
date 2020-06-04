import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])

    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # raise NotImplementedError
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            files[f] = contents

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # raise NotImplementedError
    # Extract words
    document = document.lower().translate(str.maketrans('', '', string.punctuation))
    contents = [
        # All words in the returned list should be lowercased.
        word.lower() for word in
        nltk.word_tokenize(document)
        # Filter out punctuation and stopwords
        if word not in nltk.corpus.stopwords.words("english")
    ]
    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # raise NotImplementedError
    idfs = dict()
    # create set of words
    words = set()
    for filename in documents:
        for word in documents[filename]:
            words.add(word)

    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # raise NotImplementedError
    tfidfs = dict()
    for filename in files:
        tfidfs[filename] = 0
        for word in query:
            tf = files[filename].count(word)
            tfidfs[filename] += tf * idfs[word]

    # Sort and get top n TF-IDFs for each file
    result = sorted(tfidfs.keys(), key=lambda x: tfidfs[x], reverse=True)
    result = list(result)
    return result[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # raise NotImplementedError
    sentence_tfidfs = dict()
    for sentence in sentences:
        words = sentences[sentence]
        total_idf = 0
        count = 0
        for word in query:
            if word in words:
                # Sentences should be ranked according to “matching word measure”: namely,
                # the sum of IDF values for any word in the query that also appears in the sentence.
                total_idf += idfs[word]
                count += 1
        # Query term density is defined as the proportion of words in the sentence that are also words in the query.
        sentence_tfidfs[sentence] = (total_idf, count/len(words))

    # Sort and get top n TF-IDFs for each file
    result = sorted(sentence_tfidfs.keys(), key=lambda x: sentence_tfidfs[x], reverse=True)
    result = list(result)
    return result[:n]


if __name__ == "__main__":
    main()
