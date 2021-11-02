from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
from os import path
from math import log

from typing import Union, Dict

from sklearn.feature_extraction.text import TfidfVectorizer

from qanta_util.qbdata import QantaDatabase
from tfidf_guesser_test import StubDatabase

from sgd import kBIAS

import requests

MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


class TfidfGuesser:
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """

        # You may want to add addtional data members
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None #sol

    def train(self, training_data: Union[StubDatabase, QantaDatabase], limit=-1) -> None:
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """
        
        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        # Your code here
        self.i_to_ans = {i: ans for i, ans in enumerate(answers)} #sol
        self.tfidf_vectorizer = TfidfVectorizer( #sol
            ngram_range=(1, 3), min_df=2, max_df=.9 #sol
        ).fit(questions) #sol
        self.tfidf_matrix = self.tfidf_vectorizer.transform(questions) #sol

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """

        guesses = []
        representations = self.tfidf_vectorizer.transform(questions)  #sol
        guess_matrix = self.tfidf_matrix.dot(representations.T).T     #sol
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]   #sol
        guesses = []   #sol
        for i in range(len(questions)):   #sol
            idxs = guess_indices[i]   #sol
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])   #sol

        return guesses

    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param evaluation_data: Database of questions and answers
        :param limit: How many evaluation questions to use
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the guess
        # function.

        questions = [x.text for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]
        
        d = defaultdict(dict)
        data_index = 0 #sol
        guesses = [x[0][0] for x in self.guess(questions, max_n_guesses=1)] #sol
        for gg, yy in zip(guesses, answers): #sol
            d[yy][gg] = d[yy].get(gg, 0) + 1 #sol
            data_index += 1 #sol
            if data_index % 100 == 0: #sol
                print("%i/%i for confusion matrix" % (data_index, #sol
                                                      len(guesses))) #sol
        return d
    
    def save(self): #sol
        with open(MODEL_PATH, 'wb') as f: #sol
            pickle.dump({ #sol
                'i_to_ans': self.i_to_ans, #sol
                'tfidf_vectorizer': self.tfidf_vectorizer, #sol
                'tfidf_matrix': self.tfidf_matrix #sol
            }, f) #sol

    @classmethod #sol
    def load(self): #sol
        """
        Load the guesser from a saved file
        """
        
        with open(MODEL_PATH, 'rb') as f: #sol
            params = pickle.load(f) #sol
            guesser = TfidfGuesser() #sol
            guesser.tfidf_vectorizer = params['tfidf_vectorizer'] #sol
            guesser.tfidf_matrix = params['tfidf_matrix'] #sol
            guesser.i_to_ans = params['i_to_ans'] #sol
            return guesser #sol

def get_links(page):
    session = requests.Session()

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": page,
        "prop": "links",
        "pllimit": "max",
        "plnamespace": "0"
    }

    r = session.get(url=url, params=params)
    data = r.json()
    pages = data["query"]["pages"]

    page_titles = []

    for key, val in pages.items():
        for link in val["links"]:
            page_titles.append(link["title"])

    while "continue" in data:
        plcontinue = data["continue"]["plcontinue"]
        params["plcontinue"] = plcontinue

        try:
            response = session.get(url=url, params=params)      # try/catch
            data = response.json()
            pages = data["query"]["pages"]

            for key, val in pages.items():
                for link in val["links"]:
                    page_titles.append(link["title"])
        except ConnectionResetError as e:
            print(e)

    return page_titles

def links_feature(page, question):
    page_titles = get_links(page)
    lf = 0
    for t in page_titles:
        lf += t in question
    return lf, len(page_titles)

def disambig_feature(page, question):
    if '(' in page and ')' in page:
        return page[page.find("(")+1:page.find(")")] in question
    return False

# You won't need this for this homework, but it will generate the data for a
# future homework; included for reference.
def write_guess_json(guesser, filename, fold, run_length=200, censor_features=["id", "label"]):
    """
    Returns the vocab, which is a list of all features.
    """

    vocab = [kBIAS]

    print("Writing guesses to %s" % filename)
    num = 0
    with open(filename, 'w') as outfile:
        total = len(fold)
        for qq in fold:
            num += 1
            if num % (total // 80) == 0:
                print('.', end='', flush=True)
            
            runs = qq.runs(run_length)
            guesses = guesser.guess(runs[0], max_n_guesses=10)

            for raw_guess, rr in zip(guesses, runs[0]):
                gg, ss = raw_guess[0]
                lf, lfn = links_feature(gg, rr)

                guess = {
                        "id": qq.qanta_id,
                        "guess:%s" % gg: 1,
                        "run_length": len(rr)/1000,
                        "score": ss,
                        "label": qq.page==gg,
                        "category:%s" % qq.category: 1,
                        "year:%s" % qq.year: 1,
                        # NEW FEATURES
                        "disambig": disambig_feature(gg, rr),
                        "multiguess": [x[0] for x in raw_guess].count(gg),
                        "scoreXrlength": (len(rr)/1000)*ss,
                        "cat_geo": qq.category=='Geography',
                        "links": lf,
                        "num_links": lfn
                        }

                for ii in guess:
                    # Don't let it use features that would allow cheating
                    if ii not in censor_features and ii not in vocab:
                        vocab.append(ii)

                outfile.write(json.dumps(guess, sort_keys=True))
                outfile.write("\n")
    print("")
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="data/small.guesstrain.json", type=str)
    parser.add_argument("--guessdev", default="data/small.guessdev.json", type=str)
    parser.add_argument("--buzztrain", default="data/small.buzztrain.json", type=str)
    parser.add_argument("--buzzdev", default="data/small.buzzdev.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--vocab", default="small_guess.vocab", type=str)
    parser.add_argument("--buzztrain_predictions", default="small_guess.buzztrain.jsonl", type=str)
    parser.add_argument("--buzzdev_predictions", default="small_guess.buzztest.jsonl", type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    guessdev = QantaDatabase(flags.guessdev)

    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(guesstrain, limit=flags.limit)
    tfidf_guesser.save() #sol

    confusion = tfidf_guesser.confusion_matrix(guessdev, limit=-1)
    print("Errors:\n=================================================")
    for ii in confusion:
        for jj in confusion[ii]:
            if ii != jj:
                print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))

    if flags.buzztrain_predictions:
        print("Loading %s" % flags.buzztrain)
        buzztrain = QantaDatabase(flags.buzztrain)        
        vocab = write_guess_json(tfidf_guesser, flags.buzztrain_predictions, buzztrain.buzz_train_questions)

    if flags.vocab:
        with open(flags.vocab, 'w') as outfile:
            for ii in vocab:
                outfile.write("%s\n" % ii)

    if flags.buzzdev_predictions:
        assert flags.buzztrain_predictions, "Don't have vocab if you don't do buzztrain"
        print("Loading %s" % flags.buzzdev)    
        buzzdev = QantaDatabase(flags.buzzdev)
        write_guess_json(tfidf_guesser, flags.buzzdev_predictions, buzzdev.buzz_dev_questions)
    