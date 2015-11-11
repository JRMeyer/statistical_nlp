import nltk
from nltk import corpus
from nltk.tag import DefaultTagger, RegexpTagger, NgramTagger, hmm
import matplotlib.pyplot as plt


def get_train_test_sets(percentTrain, percentTest):
    sents=[]
    for sent in corpus.treebank.tagged_sents():
        sents.append(sent)

    trainSents = sents[:int(round((len(sents)*percentTrain),0))]
    testSents = sents[int(round((len(sents)*(1-percentTest)),0)):]
    return trainSents,testSents


def train_taggers(trainSents):
    hmm_tagger = hmm.HiddenMarkovModelTagger.train(trainSents)
    default_tagger = DefaultTagger('NN')
    unigram_tagger = NgramTagger(n=1,train=trainSents)
    bigram_tagger = NgramTagger(n=2,train=trainSents)
    trigram_tagger = NgramTagger(n=3,train=trainSents)
    regexp_tagger = RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'),   # articles
         (r'.*able$', 'JJ'),                # adjectives
         (r'.*ness$', 'NN'),                # nouns formed from adjectives
         (r'.*ly$', 'RB'),                  # adverbs
         (r'.*s$', 'NNS'),                  # plural nouns
         (r'.*ing$', 'VBG'),                # gerunds
         (r'.*ed$', 'VBD'),                 # past tense verbs
         (r'.*', 'NN')                      # nouns (default)
        ])

    Taggers = [default_tagger,unigram_tagger,bigram_tagger,
               trigram_tagger, regexp_tagger, hmm_tagger]

    return Taggers


def test_tagger(Tagger,testSents):
    wordError=0
    wordTotal=0
    sentenceError=0
    sentenceTotal=0
    for sent in testSents:
        curSentError=0
        sentenceTotal+=1
        predictions = (Tagger.tag([word for word,tag in sent]))
        for i,prediction in enumerate(predictions):
            wordTotal+=1
            if prediction != sent[i]:
                wordError+=1
                curSentError=1
            else:
                pass
        sentenceError+=curSentError

    print(Tagger)
    print(1-sentenceError/sentenceTotal)
    print(1-wordError/wordTotal)

    
def plot_nltk_hmm_performance():
    performances = [(1, 55.4070597837176),
                    (2, 63.85431544582738),
                    (3, 68.27178126912875),
                    (4, 70.47541318098347),
                    (5, 71.92409712303611),
                    (10, 76.60681493572741),
                    (25, 82.62599469496021),
                    (50, 86.47214854111406),
                    (75, 88.87982044480718),
                    (90, 89.92042440318302)]

    X=[]
    Y=[]
    for x,y in performances:
        X.append(x)
        Y.append(y)


    plt.plot(X,Y)
    plt.xlabel("Percent of Penn Treebank used at Train")
    plt.ylabel("Percent Word Accuracy at Test")
    plt.title("Performance of NLTK HMM POS Tagger")
    plt.show()


    
if __name__ == "__main__":
    # training=[.01, .02, .03, .04, .05, .10, .25, .50, .75, .90]
    training=[.90]
    plot_nltk_hmm_performance()
    for percentTrain in training:
        trainSents, testSents = get_train_test_sets(percentTrain,percentTest=.1)
        print(len(trainSents))
        print(len(testSents))
        Taggers = train_taggers(trainSents)
        for Tagger in Taggers:
            test_tagger(Tagger, testSents)


