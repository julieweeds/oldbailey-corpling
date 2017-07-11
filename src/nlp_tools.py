
# coding: utf-8

# # Corpus Analysis for Old Bailey Data
# 
# First import necessary packages - spacy is for nlp analysis

# In[1]:

import spacy
nlp=spacy.load('en')
from collections import defaultdict
import logging


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
        self.allworddict=defaultdict(int)
        self.worddict=defaultdict(int)
        self.noundict=defaultdict(int)
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
                    self.docs.append(line)
                
       
    
    def basic_analyse_all(self,ner=False):
        if ner:
            logging.info("Running basic analysis with NER")
        else:
            logging.info("Running basic analysis")
        todo=len(self.docs)
        if self.prop<10:
            tenpercent=((todo*self.prop)//100)+1
        else:
            tenpercent=(todo//10)+1
        logging.info("Analysing {}%. Chunks of size {}".format(self.prop,tenpercent))
        self.count=0        
        for doc in self.docs:
            nlpdoc=self.basic_analyse_single(doc)
            if ner:
                self.explore_ner(nlpdoc,self.count)
        
            if self.count%tenpercent==0:
                done=self.count*100/todo
                logging.info("Completed {} docs ({}% complete)".format(str(self.count),str(done)))
                if done >= self.prop:
                    break
                
        logging.info("Number of documents is {}".format(self.count))
        #print("Distribution of document lengths is {}".format(str(self.doclengths)))
        #print("Distribution of sentence lengths is {}".format(str(self.sentencelengths)))
        #print("Distribution of word lengths is {}".format(str(self.wordlengths)))
        #print("Number of docs with 1 sentence is {}".format(self.doclengths[1]))
        
    def basic_analyse_single(self,doc):
        
        self.count+=1
        nlpdoc=nlp(doc)
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
    
        

