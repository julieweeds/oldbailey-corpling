
# coding: utf-8

# # Corpus Analysis for Old Bailey Data
# 
# First import necessary packages - spacy is for nlp analysis, pandas and matplotlib for data analysis and visualisation

# In[1]:

import spacy
import pandas as pd
import matplotlib as plt
get_ipython().magic('matplotlib inline')
import time
#nlp=spacy.load('en')
from collections import defaultdict
import operator,math
from gensim.models import Word2Vec


# The corpus class loads in, stores and performs basic nlp analysis on a corpus.  The data structures generated during basic analysis can be accessed for further analysis and visualisation.

# In[82]:

class corpus:
    
    loctypes=["LOC","GPE","FAC"]
    
    def __init__(self,ipfiles,nlpmodel,prop=10,ner=False,loadfiles=True):
        #default mode is to load in 10%.  Set prop = 100 to load in whole corpus
        self.sourcefiles=ipfiles
        self.nlp=nlpmodel
        self.prop=prop
        self.name=""
        self.docs=[]
        self.nlpdocs=[]
        self.allworddict=defaultdict(int)
        self.worddict=defaultdict(int)
        self.noundict=defaultdict(int)
        self.propnoundict=defaultdict(int)
        self.verbdict=defaultdict(int)
        self.adjdict=defaultdict(int)
        self.advdict=defaultdict(int)
        self.wordposdict=defaultdict(int)
        self.wordlengths=defaultdict(int)
        self.sentencelengths=defaultdict(int)
        self.doclengths=defaultdict(int)
        self.sentences=[]
        self.content_sentences=[]
        self.pos_sentences=[]

        self.wordtotal=0
        self.nountotal=0
        self.propnountotal=0
        self.verbtotal=0
        self.adjtotal=0
        self.advtotal=0
        
        if loadfiles:
            self.initialise()
        else:
            self.docs=ipfiles
            self.name="unknown"
        self.basic_analyse_all(ner=ner)
        
    def initialise(self):
        print("Loading sourcefiles")
        for ipf in self.sourcefiles:
            with open(ipf) as input:
                self.name+="_"+ipf
                for line in input:
                    self.docs.append(line.rstrip())
                
       
    
    def basic_analyse_all(self,ner=False):
        if ner:
            print ("Running basic analysis with NER")
        else:
            print("Running basic analysis")
        todo=len(self.docs)
        if self.prop<10:
            tenpercent=((todo*self.prop)//100)+1
        else:
            tenpercent=(todo//10)+1
        print("Analysing {}%. Chunks of size {}".format(self.prop,tenpercent))
        self.count=0        
        for doc in self.docs:
            nlpdoc=self.basic_analyse_single(doc)
            self.nlpdocs.append(nlpdoc)
            if ner:
                self.explore_ner(nlpdoc,self.count)
        
            if self.count%tenpercent==0:
                done=self.count*100/todo
                print("Completed {} docs ({}% complete)".format(str(self.count),str(done)))
                if done >= self.prop:
                    break
                
        print("Number of documents is {}".format(self.count))
        #print("Distribution of document lengths is {}".format(str(self.doclengths)))
        #print("Distribution of sentence lengths is {}".format(str(self.sentencelengths)))
        #print("Distribution of word lengths is {}".format(str(self.wordlengths)))
        #print("Number of docs with 1 sentence is {}".format(self.doclengths[1]))
        
    def basic_analyse_single(self,doc):
        
        self.count+=1
        nlpdoc=self.nlp(doc)
        nosents=0
        for sent in nlpdoc.sents:
            sent_text=[]
            content_text=[]
            pos_text=[]
            slength=len(sent)
            #if slength>500:
            #    print(sent)
            self.sentencelengths[slength]+=1
            nosents+=1
            for token in sent:
                
                wordlength=len(token)
                self.wordlengths[wordlength]+=1
                self.wordtotal+=1
                
                self.allworddict[token.text.lower()]+=1
                self.wordposdict[(token.text.lower(),token.pos_)]+=1
                sent_text.append(token.text.lower())
                pos_text.append(token.text.lower()+"_"+token.pos_)
                if not token.is_stop and not token.is_oov and not token.pos_=="PUNCT":
                    self.worddict[token.lemma_]+=1
                    content_text.append(token.lemma_)
                    if token.pos_ =="NOUN":
                        self.noundict[token.lemma_]+=1
                        self.nountotal+=1
                    elif token.pos_ =="PROPN":
                        self.propnoundict[token.lemma_]+=1
                        self.propnountotal+=1
                    elif token.pos_=="VERB":
                        self.verbdict[token.lemma_]+=1
                        self.verbtotal+=1
                    elif token.pos_=="ADJ":
                        self.adjdict[token.lemma_]+=1
                        self.adjtotal+=1
                    elif token.pos_=="ADV":
                        self.advdict[token.lemma_]+=1
                        self.advtotal+=1
                        
            self.sentences.append(sent_text)    
            self.content_sentences.append(content_text)
            self.pos_sentences.append(pos_text)
        #print("Number of sentences is {}".format(nosents))
        self.doclengths[nosents]+=1
        return nlpdoc
        
    def get_word_distribution(self,wordtype):
        
        if wordtype=="NOUN":
            return(self.noundict,self.nountotal)
        elif wordtype=="VERB":
            return(self.verbdict,self.verbtotal)
        elif wordtype=="ADJ":
            return(self.adjdict,self.adjtotal)
        elif wordtype=="ADV":
            return(self.advdict,self.advtotal)
        else:
            return(self.worddict,self.wordtotal)

    def get_sentences(self):
        for sent in self.sentences:
            yield sent
            
            
    def explore_ner(self,doc,number):
        for sent in doc.sents:
            containsloc=False
            for token in sent:
                if token.ent_type_ in corpus.loctypes:
                    containsloc=True
            if containsloc:
                
                print("Document {}: {}".format(number,sent))
                for token in sent: 
                    if len(token.text)<8:
                        print("{}\t{}\t\t{}\t\t{}\t{}\t{}\t{}".format(token.i,token.text,token.lemma_,token.pos_,token.ent_type_,token.dep_,token.head.i))
                    else:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(token.i,token.text,token.lemma_,token.pos_,token.ent_type_,token.dep_,token.head.i))
    
        


def summarise(freqtable_dict):
    
    sumf=0
    sumxf=0
    sumxxf=0
    for key in freqtable_dict.keys():
        sumf+=freqtable_dict[key]
        sumxf+=freqtable_dict[key]*key
        sumxxf+=freqtable_dict[key]*key*key
        
    mean=sumxf/sumf
    var = sumxxf/sumf-mean*mean
    sd=math.sqrt(var)
    
    print("Mean is {} and sd is {}".format(str(mean),str(sd)))
    return(mean,sd)
    


def squash(afreqdict,m,sd):
    
    bmax=0
    threshold=math.ceil(m+sd)
    
    count=0
    bfreqdict=defaultdict(int)
    print("Threshold for inclusion in bar chart is {}".format(threshold))
    for (key,value) in afreqdict.items():
        #print(key)
        #print(type(key))
        if key>threshold:
            count+=value
        else:
            bfreqdict[key]=value
            
        if key > bmax:
            bmax=key
    #print(count)
    label=threshold
    bfreqdict[label]=count
    return bfreqdict



def visualise(afreqdict,heading='',makesquash=True):

    (m,sd)=summarise(afreqdict)
    if makesquash:
        bfreqdict=squash(afreqdict,m,sd)
    else:
        bfreqdict=afreqdict
    docsdata = pd.DataFrame.from_dict(bfreqdict,orient='index')
    docsdata.sort_index(inplace=True)
    docsdata.plot.bar(title='Distribution of number of '+heading,legend=False,figsize=(12,6))
    return docsdata



# ## Popular Words
# 
# Find words (possibly of given part of speech) which are most common in a given corpus.  Then find the ones which are 'most surprising' given a second comparative corpus.

# In[57]:

#import operator

def find_most_common_words(corpus1,wordtype,n=10):

    (dist1,_)=corpus1.get_word_distribution(wordtype)
    dc_sort = sorted(dist1.items(),key = operator.itemgetter(1),reverse = True)
    return dc_sort[0:n]



def find_surprising_words(corpus1,corpus2,wordtype,n=10,shift=0.25):
    #will find words which are surprisingly frequent in dist1 given dist2
    
    
    (dist1,total1)=corpus1.get_word_distribution(wordtype)
    (dist2,total2)=corpus2.get_word_distribution(wordtype)
    
    candidates={}
   
    
    for(key,value) in dist1.items():
        if value>0:
            p1=value/total1
            p2=(dist2[key]+0.1)/(total2+0.1)
            #p2=(dist2[key]+value)/(total2+total1)
            
            llr=math.log(p1/p2)
        
            if llr>0:
                
                llr=(1-shift)*llr+math.log(p1)
                candidates[key]=llr

       
    dc_sort=sorted(candidates.items(),key=operator.itemgetter(1),reverse=True)
    freqs=[(key,dist1[key]) for (key,_) in dc_sort[0:n]]
    
    return freqs
   

# In[ ]:



