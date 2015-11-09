import nltk
from nltk import corpus
from nltk.tag import DefaultTagger, RegexpTagger, NgramTagger


def get_train_test_sets(percentTrain):
    sents=[]
    for sent in corpus.treebank.tagged_sents():
        sents.append(sent)

    trainSents = sents[:int(round((len(sents)*percentTrain),0))]
    testSents = sents[int(round((len(sents)*percentTrain),0)):]
    return trainSents,testSents


def train_taggers(trainSents):
    default_tagger = DefaultTagger('NN')
    unigram_tagger = NgramTagger(n=1,train=list(corpus.treebank.tagged_sents()))
    bigram_tagger = NgramTagger(n=2,train=list(corpus.treebank.tagged_sents()))
    trigram_tagger = NgramTagger(n=3,train=list(corpus.treebank.tagged_sents()))
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
               trigram_tagger, regexp_tagger]
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


if __name__ == "__main__":
    trainSents, testSents = get_train_test_sets(percentTrain=.9)
    Taggers = train_taggers(trainSents)
    for Tagger in Taggers:
        test_tagger(Tagger, testSents)
