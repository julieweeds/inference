####
#implementation of Pilehvar and Collier 2017
#Correlation of similarity judgments with human judgments for rare words
#To obtain similar results to them, need case_insensitive=True and unknown=random (or dummy)
#Still find an extra 2 pairs which are not OOV (also found by the gensim implementation which runs via WordReps.correlate()
#The idea is to look at different ways of finding semantic landmarks including the distributional inclusion hypothesis ... not sure that this will work for dense representations
#maybe experiment between centroid, max and min?
####


import os,sys,configparser,csv, numpy as np, scipy.stats as stats,math,ast
from gensim.models import KeyedVectors
from gensim import matutils
from nltk.corpus import wordnet as wn

def sim(vectorA,vectorB,metric="dot"):

    if metric=="cosine":
        #num=np.dot(vectorA,vectorB)
        #den=math.pow(np.dot(vectorA,vectorA)*np.dot(vectorB,vectorB),0.5)
        #return num/den
        return np.dot(matutils.unitvec(vectorA),matutils.unitvec(vectorB))
    else:
        return np.dot(vectorA,vectorB)

def getrank(wordvectors,word):
    #not used ... use WordReps.wv.index2word instead
    try:
        rank= wordvectors.vocab[word].__dict__['index']
    except:
        rank=-1
    return rank

def find_landmarks(word,vocab=[],method=["synonyms"],case_insensitive = False):
    if case_insensitive:
        wnsynsets = wn.synsets(word.lower())
        wnlemmas = wn.lemmas(word.lower())
    else:
        wnsynsets = wn.synsets(word)
        wnlemmas = wn.lemmas(word)

    landmarks=[]
    for m in method:

        if m=="synonyms":
            landmarks+=find_synonyms(wnsynsets)
        elif m =="antonyms":
            landmarks+=find_antonyms(wnlemmas)
        else:
            landmarks+=find_related(wnsynsets,relation=m)

    if case_insensitive:
        landmarks = [str(lemma.name()).upper() for lemma in landmarks]
    else:
        landmarks = [str(lemma.name()) for lemma in landmarks]

    candidates=[]
    for landmark in landmarks:
        if landmark != word:
            if vocab==[] or landmark in vocab:
                candidates.append(landmark)
    #print(word,candidates)
    return candidates

def find_synonyms(wnsynsets):
    landmarks = []
    for wnsynset in wnsynsets:
        landmarks += wnsynset.lemmas()

    return landmarks

def find_antonyms(wnlemmas):
    landmarks=[]
    for l in wnlemmas:
        landmarks+=l.antonyms()
    return landmarks

def hypernyms(wnsynset):
    return wnsynset.hypernyms()+wnsynset.instance_hypernyms()

def ancestors(wnsynset):
    hs=hypernyms(wnsynset)
    ancs=hs
    for hypernym in hs:
        ancs+=ancestors(hypernym)
    return ancs

def find_related(wnsynsets,relation="hypernyms"):
    landmarks=[]
    for wnsynset in wnsynsets:
        if relation =="ancestors":
            related=ancestors(wnsynset)

        elif relation =="hypernyms":
            related = hypernyms(wnsynset)
        elif relation == "hyponyms":
            related = wnsynset.hyponyms()+wnsynset.instance_hyponyms()
        elif relation == "meronyms":
            related = wnsynset.member_meronyms()+wnsynset.substance_meronyms()+wnsynset.part_meronyms()
        elif relation == "holonyms":
            related = wnsynset.member_holonyms()+wnsynset.substance_holonyms()+wnsynset.part_holonyms()
        elif relation == "attributes":
            related = wnsynset.attributes()
        elif relation == "entailments":
            related = wnsynset.entailments()
        elif relation == "cohyponyms" or relation == "co-hyponyms":
            hypernyms=wnsynset.hypernyms()+wnsynset.instance_hypernyms()

            related=[]
            for hypernym in hypernyms:
                cohyponyms=hypernym.hyponyms()+hypernym.instance_hyponyms()
                for cohyponym in cohyponyms:
                    if cohyponym!=wnsynset:
                        related.append(cohyponym)
        elif relation == "grandparents":
            hypernyms=hypernyms(wnsynset)
            related=[]
            for hypernym in hypernyms:
                related+=hypernyms(hypernym)


        else:
            print("Unknown method / relation {}".format(relation))
            related=[]
        for w in related:
            landmarks+=w.lemmas()
    return landmarks[:25]

class WordReps:

    def __init__(self,vectorfile,vocabsize,unknown='ignore',case_insensitive=True):
        print("Initialising WordReps with {}".format(vectorfile))
        self.wv=KeyedVectors.load_word2vec_format(vectorfile,binary=True)
        self.vectorcache={}
        self.vocabsize=vocabsize
        self.unknown=unknown
        self.UNK = np.random.uniform(-0.25, 0.25, 300)
        self.case_insensitive=case_insensitive
        self.ok_vocab=[(w,w) for w in self.wv.index2word[0:self.vocabsize]]
        if self.case_insensitive:
            self.ok_vocab=[(x.upper(),y) for (x,y) in self.ok_vocab]
        self.ok_vocab=dict(reversed(self.ok_vocab))

    def correlate(self,datafile):
        corr=self.wv.evaluate_word_pairs(datafile,restrict_vocab=self.vocabsize,case_insensitive=self.case_insensitive,dummy4unknown=(self.unknown=='dummy'))
        print("Correlation with {} is {}".format(datafile,corr))

    def makecache(self,wordlist):
        self.vectorcache={}
        if self.case_insensitive:
            wordlist=[word.upper() for word in wordlist]

        self.oovwords=[]
        for word in wordlist:
            if word in self.ok_vocab.keys():
                self.vectorcache[word]=self.wv[self.ok_vocab[word]]
            else:
                if self.unknown=='random-diff':
                    self.vectorcache[word] = np.random.uniform(-0.25, 0.25, 300)
                else:
                    self.vectorcache[word]=self.UNK

                self.oovwords.append(word)

        percent=len(self.oovwords)*100.0/len(wordlist)
        print("{} OOV words ({}\%)".format(len(self.oovwords),percent))
        print(self.vectorcache.keys())

    def update_one(self,word,landmarks,theta=0):
        result=theta * self.vectorcache[word]
        #print(result)
        n=len(landmarks)
        for index,landmark in enumerate(landmarks):
            i=index+1
            multiplier= math.exp(-i)/n
            result+=multiplier * self.wv[self.ok_vocab[landmark]]
            #print(multiplier, self.vectorcache[word])
            #break

        #print(result)
        return result

    def infer(self,flag='oov',method="synonyms"):
        #inference for just oovwords or for all words
        if flag == 'all':
            words=self.vectorcache.keys()

        else:
            words=self.oovwords

        landmarks={}
        for word in words:
            landmarks[word]=find_landmarks(word,self.ok_vocab.keys(),method=method,case_insensitive=self.case_insensitive)

        old_oov=self.oovwords
        self.oovwords=[]
        for word in words:
            if word in old_oov and len(landmarks[word])==0:
                self.oovwords.append(word)
            else:
                if word in old_oov:
                    theta=0
                else:
                    theta=1
                self.vectorcache[word]=self.update_one(word,landmarks[word],theta=theta)
                #break

        #sys.exit()

    def randomise(self,words):
        self.vectorcache={}
        self.oovwords=[]
        if self.case_insensitive:
            words=[word.upper() for word in words]

        #shift word embeddings by 1 in the list
        if words[0] in self.ok_vocab.keys():
            first=self.wv[self.ok_vocab[words[0]]]
        else:
            if self.unknown=='random-diff':
                first=np.random.uniform(-0.25,0.25,300)
            else:
                first = self.UNK
            self.oovwords.append(words[-1])
        for i,word in enumerate(words):
            if i<len(words)-1:
                if words[i+1] in self.ok_vocab.keys():
                    self.vectorcache[word]=self.wv[self.ok_vocab[words[i+1]]]
                else:
                    if self.unknown=='random-diff':
                        self.vectorcache[word]=np.random.uniform(-0.25,0.25,300)
                    else:
                        self.vectorcache[word]=self.UNK
                    self.oovwords.append(word)
            else:
                self.vectorcache[word]=first

    def correlate_cache(self,dataset):

        xs=[]
        ys=[]
        oovpair=0
        for tuple in dataset.tuples:
            if self.case_insensitive:
                wordA=tuple[0].upper()
                wordB=tuple[1].upper()
            else:
                wordA=tuple[0]
                wordB=tuple[1]
            asim = sim(self.vectorcache[wordA], self.vectorcache[wordB], metric="cosine")
            if wordA in self.oovwords or wordB in self.oovwords:
                oovpair+=1
                if self.unknown.startswith("random"):
                    xs.append(float(tuple[2]))
                    ys.append(asim)
                elif self.unknown=="dummy":
                    xs.append(float(tuple[2]))
                    ys.append(0)
                else:
                    continue

            else:
                xs.append(float(tuple[2]))
                ys.append(asim)
            #print("sim({}, {}) = {} cf {}".format(tuple[0],tuple[1],asim,tuple[2]))


        xarray=np.asarray(xs)
        yarray=np.asarray(ys)

        percent=oovpair*100/len(dataset.tuples)
        print("{} OOV pairs ({}\%)".format(oovpair,percent))
        print("{} pairs for correlation".format(len(xs)))

        rho=stats.spearmanr(xarray,yarray)
        print("SpearmanrResult for cache is {}".format(rho))



    def test(self):
        asim=self.wv.similarity('woman','man')
        print("Similarity between woman and man is {}".format(asim))
        #print(self.wv['man'])
        #womanprob=self.model.score(["woman".split()])
        #ladyprob=self.model.score(["lady".split()])
        #print("P(woman) = {} and P(lady) = {}".format(womanprob,ladyprob))
        print(self.wv.vocab['woman'])
        woman=self.wv.vocab['woman']
        print(woman.__dict__)
        print(woman.__dict__['index'])
        if self.wv.vocab['woman'].__lt__(self.wv.vocab['lady']):print("woman")
        if self.wv.vocab['lady'].__lt__(self.wv.vocab['woman']):print("lady")
        print(getrank(self.wv,'man'))

#        ws353corr=self.wv.evaluate_word_pairs(os.path.join('~/Documents/workspace/datasets','wordsim353.tsv'))
#        print("Correlation with WordSim353 is {}".format(ws353corr))



class Dataset:

    def __init__(self,datafile,header=0):
        print("Loading dataset from {}".format(datafile))
        self.tuples=[]
        with open(datafile,'r') as tsvin:

            tsvreader=csv.reader(tsvin,delimiter='\t')
            for i,row in enumerate(tsvreader):
                if i >header:
                    self.tuples.append(row[0:3])
        self.words_of_interest=[]


    def rewrite(self,outfile):
        with open(outfile,'w') as tsvout:
            tsvwriter=csv.writer(tsvout,delimiter='\t')
            for tuple in self.tuples:
                tsvwriter.writerow(tuple)



    def get_words_of_interest(self):
        #print(self.tuples)
        if self.words_of_interest==[]:

            for tuple in self.tuples:
                if tuple[0] not in self.words_of_interest:
                    self.words_of_interest.append(tuple[0])
                if tuple[1] not in self.words_of_interest:
                    self.words_of_interest.append(tuple[1])

        print("Number of pairs in dataset is {}, number of words of interest is {}".format(len(self.tuples),len(self.words_of_interest)))
        #print(self.words_of_interest)
        return self.words_of_interest



    def test(self):
        print(len(self.tuples))

class Correlator:

    def __init__(self,configfile):
        print("Initialising correlator with {}".format(configfile))
        self.config=configparser.ConfigParser()
        self.config.read(configfile)

        self.testing=self.config.getboolean('default','testing')
        dataset=self.config.get('default','dataset')
        self.datafile=os.path.join(self.config.get('default','datapath'),self.config.get(dataset,'datafile'))


        #self.myData.rewrite(self.datafile+".new")

        self.vectorfile = os.path.join(self.config.get('default', 'vectorpath'),
                                       self.config.get('default', 'vectorfile'))
        self.vocabsize=int(self.config.get('default','vocabsize'))
        self.unknown = self.config.get('default', 'unknown')
        self.case_insensitive= self.config.getboolean('default','case_insensitive')
        self.infer_flag=self.config.get('default','infer_flag')
        self.infer_method=ast.literal_eval(self.config.get('default','infer_method'))

        if not self.testing:
            self.myData = Dataset(self.datafile, header=int(self.config.get(dataset, 'header')))
            self.myWordReps = WordReps(self.vectorfile,vocabsize=self.vocabsize,unknown=self.unknown,case_insensitive=self.case_insensitive)


    def test(self):
        #self.myWordReps.test()
        #self.myData.test()
        words=["TRICLINIC","SQUIRT","TREE","DRY","WALK"]
        for word in words:
            print(word)

            landmarks=find_landmarks(word,method=self.infer_method,case_insensitive=self.case_insensitive)

            print(landmarks)


    def run(self):
        if self.testing:
            self.test()
        else:
            self.myWordReps.correlate(self.datafile)
            words=self.myData.get_words_of_interest()
            self.myWordReps.makecache(words)
            self.myWordReps.correlate_cache(self.myData)
            #self.myWordReps.randomise(words)
            #self.myWordReps.correlate_cache(self.myData)

            self.myWordReps.infer(flag=self.infer_flag,method=self.infer_method)
            self.myWordReps.correlate_cache(self.myData)

if __name__=="__main__":

    if len(sys.argv)>1:
        configfile=sys.argv[1]
    else:
        configfile="config.cfg"

    myCorrelator=Correlator(configfile)
    myCorrelator.run()
