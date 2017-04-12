import os,sys,configparser,csv, numpy as np, scipy.stats as stats,math
from gensim.models import KeyedVectors
from gensim import matutils

def sim(vectorA,vectorB,metric="dot"):

    if metric=="cosine":
        #num=np.dot(vectorA,vectorB)
        #den=math.pow(np.dot(vectorA,vectorA)*np.dot(vectorB,vectorB),0.5)
        #return num/den
        return np.dot(matutils.unitvec(vectorA),matutils.unitvec(vectorB))
    else:
        return np.dot(vectorA,vectorB)

def getrank(wordvectors,word):

    try:
        rank= wordvectors.vocab[word].__dict__['index']
    except:
        rank=-1
    return rank



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
        #print(self.vectorcache.keys())

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


        dataset=self.config.get('default','dataset')
        self.datafile=os.path.join(self.config.get('default','datapath'),self.config.get(dataset,'datafile'))
        self.myData=Dataset(self.datafile,header=int(self.config.get(dataset,'header')))

        #self.myData.rewrite(self.datafile+".new")

        self.vectorfile = os.path.join(self.config.get('default', 'vectorpath'),
                                       self.config.get('default', 'vectorfile'))
        self.vocabsize=int(self.config.get('default','vocabsize'))
        self.unknown = self.config.get('default', 'unknown')
        self.case_insensitive= self.config.getboolean('default','case_insensitive')
        self.myWordReps = WordReps(self.vectorfile,vocabsize=self.vocabsize,unknown=self.unknown,case_insensitive=self.case_insensitive)


    def test(self):
        self.myWordReps.test()
        self.myData.test()

    def run(self):
        self.myWordReps.correlate(self.datafile)
        words=self.myData.get_words_of_interest()
        #words=filter(self.myWordReps.wv,words,self.vocabsize)
        #print("Number of filtered words is {}".format(len(words)))
        self.myWordReps.makecache(words)
        self.myWordReps.correlate_cache(self.myData)
        self.myWordReps.randomise(words)
        self.myWordReps.correlate_cache(self.myData)



if __name__=="__main__":

    if len(sys.argv)>1:
        configfile=sys.argv[1]
    else:
        configfile="config.cfg"

    myCorrelator=Correlator(configfile)
    myCorrelator.run()
