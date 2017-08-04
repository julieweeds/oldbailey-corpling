import pandas as pd
import numpy as np
import matplotlib as plt
from collections import defaultdict
import random
import nlp_tools
import spacy,operator
import logging
import configparser,ast,os,sys
from time import time
import CharacterisingFunctions as cf
import csv


def process_header(header):
    # print(header)
    mydict = {}
    for i, text in enumerate(header):
        mydict[text] = i
    return mydict


def insert_space(line, currtext, respace=True):
    needspace = True
    if respace:
        if line['LEMMA'] == 'PUNC':
            if line['vard'] == '-' and currtext['LEMMA'] == 'PUNC':
                needspace = True
            else:
                needspace = False
        if line['POS'] == 'GE':
            needspace = False
        if line['vard'] == 'S_END' or line['vard'] == 'S_BEGIN':
            needspace = False
        if currtext['vard'] == '-':
            needspace = False
    return needspace


def do_include_text(line):
    needtext = True
    if line['vard'] == 'S_BEGIN' or line['vard'] == 'S_END':
        needtext = False
    return needtext


class Samuels:
    alias = {'q#TOTEN': 'vard', '#TOTEN': 'vard'}
    semtag_split={'SEMTAG1':' ','SEMTAG2':';','SEMTAG3':';'}

    def __init__(self, filepaths, prop=10, chunksize=10,lowercase=True,nlp=None,semtag_first=True):
        self.filepaths = filepaths
        self.prop = 10
        self.chunksize = 10
        self.lowercase=lowercase
        self.semtag_first=semtag_first

        self.rows = []
        self.cols = ['id', 'key']
        if self.lowercase:
            self.cols.append('vard_lower')

        self.nlp=nlp

        self.loadfile()

    def reverse_header(self, header):
        mydict = {}
        for i, text in enumerate(header):
            mydict[i] = Samuels.alias.get(text, text)
        return mydict

    def addtocols(self, values):
        for value in values:
            if value not in self.cols:
                self.cols.append(value)

    def update_semtags(self, row):
        newrow = {}
        for key in row.keys():
            if key in Samuels.semtag_split.keys():
                splitchar = Samuels.semtag_split[key]
                field = str(row[key])
                parts = field.split(splitchar)
                newrow[key] = parts[0]
            else:
                newrow[key] = row[key]
        return newrow

    def loadfile(self):
        self.errors = []
        for filepath in self.filepaths:
            logging.info("Reading {}".format(filepath))
            with open(filepath) as instream:
                newfile = True
                self.sentences = -1
                insent=-1
                for count, line in enumerate(instream):
                    row = {}
                    line = line.rstrip()
                    parts = line.split('\t')
                    if newfile:
                        headerdict = self.reverse_header(parts)
                        self.addtocols(headerdict.values())
                        newfile = False
                    elif len(parts) == 7:  # well-formed line
                        insent+=1
                        for i, field in enumerate(parts):
                            row[headerdict[i]] = field
                        # print(row)
                        if row['vard'] == 'S_BEGIN':
                            self.sentences += 1
                            insent=-1
                        if self.lowercase:
                            row['vard_lower']=row['vard'].lower()
                        row['insent']=insent
                        row['id'] = count
                        row['sentence'] = self.sentences
                        row['chunk'] = int(self.sentences / self.chunksize)
                        row['key']="{}:{}:{}".format(row['chunk'],row['sentence'],row['insent'])
                        if self.semtag_first:
                            row=self.update_semtags(row)
                        self.rows.append(row)
                    else:
                        self.errors.append(line)

        logging.info("Read {}, errors = {}".format(count, len(self.errors)))
        self.chunks = int(self.sentences / self.chunksize) + 1
        logging.info("{} chunks of sentences".format(self.chunks))

    def get_dataframe(self):

        try:
            return self.dataframe
        except:
            self.dataframe = pd.DataFrame.from_records(self.rows, columns=self.cols)
            return self.dataframe

    def make_corpus(self, field='vard', respace=True, test=False):

        docs = []
        text = ''
        newline = True
        docbuffer = 0
        currtext = self.rows[-1]
        for i, line in enumerate(self.rows):
            # print("{}:{}".format(line[header_dict['vard']],line[header_dict['pos']]))
            currdoc = line['chunk']
            if currdoc != docbuffer:
                newline = True
                docs.append(text)
                text = ''
                docbuffer = currdoc
            donespace = False
            if insert_space(line, currtext, respace=respace) and newline == False:
                text += " "
                donespace = True
            if do_include_text(line):
                currtext = line
                text += currtext[field]
                if donespace and currtext['vard'] == '-':
                    currtext['vard'] = 'MMMDASH'
                newline = False

            if i > 1000 and test:
                break
        docs.append(text)
        return docs

    def make_bow(self, field='vard', k=100000,bootstrap=False,params={}):
        # turn corpus into a bag of words for a certain field - variant of make_hfw_dist()

        sumdict = {}
        corpussize = 0
        if bootstrap:
            df =self.get_bootstrap(prop=params.get('prop',100),size=params.get('size',0))
        else:
            df = self.get_dataframe()
        df = df[df['LEMMA'] != 'NULL']
        if self.lowercase and field=='vard':
            field='vard_lower'
        for item in df[field]:
            sumdict[item] = sumdict.get(item, 0) + 1
            corpussize += 1

        logging.info("Size of corpus is {}".format(corpussize))
        candidates = sorted(sumdict.items(), key=operator.itemgetter(1), reverse=True)
        return corpussize, candidates[:k]

    def find_text(self, semtag, field='SEMTAG3'):

        df = self.get_dataframe()
        df = df[df['LEMMA'] != 'NULL']
        if self.lowercase:
            groupby='vard_lower'
        else:
            groupby='vard'
        mylemmas = df[df[field] == semtag].groupby(groupby)['id'].nunique()
        mylemmas = mylemmas.sort_values(ascending=False)
        mylist = list(mylemmas[0:10].index.values)
        return mylist

    def get_bootstrap(self,prop=100,size=0):

        N=self.chunks
        df=self.get_dataframe()
        N=int(N*prop/100)
        cont=True

        i=0
        while cont:

            chosenchunk=random.randint(0,N-1)
            if i==0:
                self.bootstrap=df[df['chunk']==chosenchunk]
            else:
                self.bootstrap=self.bootstrap.append(df[df['chunk']==chosenchunk])
            #print(len(df), len(self.bootstrap))
            i+=1
            #print(chosenchunk)
            if i > N or (size > 0 and len(self.bootstrap) >= size):  # under-sample so corpus is not bigger than specified size, no over-sampling
                cont = False
        return self.bootstrap

def make_dict(distA):
    size,alist=distA
    adict={}
    for(word,freq) in alist:
        adict[word]=freq
    return size,adict


def compare(distA,distB,indicatordict):
    #distA is a pair of size and list of tuples (word,frequencies)
    #distB is a pair of size and lookup dict

    sizeA, hfwA = distA
    sizeB, hfwBdict = distB

    for (word, freqA) in hfwA:
        freqB = hfwBdict.get(word, 0)
        probA = freqA / sizeA
        probB = freqB / sizeB
        if probA > probB:
            indicatordict[word] = indicatordict.get(word, 0) + 1
    return indicatordict

def check_convergence(newdict,cache,outfile,N,t=0.9):

    candidates = [(term, (value + 1) / (N + 1)) for (term, value) in newdict.items()]
    sortedlist = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    newcache={}
    with open(outfile,"w") as outstream:
        for (term, score) in sortedlist:
            if score < 0.505:
                break
            else:
                outstream.write("{}\t{}\n".format(term, score))
                newcache[term]=score

    absdiffs=[]
    for term in newcache.keys():
        if newcache[term]>t:
            absdiffs.append(abs(newcache[term]-cache.get(term,0)))
    for term in cache.keys():
        if cache[term]>t:
            absdiffs.append(abs(newcache.get(term,0)-cache[term]))
    #logging.info(absdiffs)
    if len(absdiffs)>0:
        cvscore=np.percentile(absdiffs,99)
    else:
        cvscore=1
    logging.info("99% convergence level at {} = {}".format(N,cvscore))
    if cvscore<0.005:
        return newcache,True
    else:
        return newcache,False




def bootstrap_compare(samA,samB,repeats=10,prop=100,interval=100,field='vard',outfile_stem='words'):

    logging.info("Generating corpus B distribution")
    distB=make_dict(samB.make_bow(field=field))
    sizeB=distB[0]
    indicatordict={}
    cacheddict={}
    N=repeats
    for j in range(0,repeats):
        logging.info("Bootstrapping corpus A repetition {}".format(j))
        distA=samA.make_bow(bootstrap=True,field=field,params={'prop':prop,'size':sizeB})

        indicatordict=compare(distA,distB,indicatordict)

        if (j+1)%interval==0:
            cacheddict,stop=check_convergence(indicatordict,cacheddict,outfile_stem+"_"+str(j+1),j+1)

            if stop:
                N=j+1
                break

    logging.info("Generating candidates")
    candidates = [(term, (value + 1) / (N + 1)) for (term, value) in indicatordict.items()]
    sortedlist = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    return sortedlist


if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)

    testing=False
    if testing:
        female = ['corpus_1800_1820_theft_f_def', 'corpus_1800_1820_theft_f_wv']
        male = ['corpus_1800_1820_theft_m_def', 'corpus_1800_1820_theft_m_wv']

        parentdir = "/Users/juliewe/Dropbox/oldbailey/speech_corpora/theft/all"
        samuels = "samuels_tagged"
        affix = "_semtagged"
        samuels_female_paths = []
        for name in female:
            samuels_female_paths.append(os.path.join(parentdir, samuels, name + affix))
        samuels_male_paths = []
        for name in male:
            samuels_male_paths.append(os.path.join(parentdir, samuels, name + affix))

        #------test 1

        fdefpath = samuels_female_paths[0]
        fdef = Samuels([fdefpath])
        #print(fdef.get_dataframe().head(10))

        #------test 2
        #bs=fdef.get_bootstrap(prop=10)
        #print(bs['vard'].head())

        #------test 3
        #bow=fdef.make_bow(field='SEMTAG1')
        #print(bow)

#        bbow=fdef.make_bow(field='SEMTAG1',bootstrap=True)
#        print(bbow)

        #----test 4
        mwvpath = samuels_male_paths[0]
        mwv=Samuels([mwvpath])

        cands=bootstrap_compare(fdef,mwv)
        print(cands)

    else:

        myconfig = configparser.ConfigParser()
        if len(sys.argv) > 1:
            configfile = sys.argv[1]
        else:
            configfile = 'samuelsbootstrap.cfg'
        myconfig.read(configfile)

        parentdir=myconfig.get('default','parentdir')
        Afiles = ast.literal_eval(myconfig.get('default', 'Afiles'))
        Bfiles = ast.literal_eval(myconfig.get('default', 'Bfiles'))
        field=myconfig.get('default','field')
        outfile=myconfig.get('default','outfile')+"_"+field


        Apaths=[os.path.join(parentdir,f) for f in Afiles]
        Bpaths=[os.path.join(parentdir,f) for f in Bfiles]

        samA=Samuels(Apaths)
        samB=Samuels(Bpaths)


        candidates = bootstrap_compare(samA,samB,field=field,repeats=myconfig.getint('default', 'repeats'),prop=myconfig.getint('default', 'prop'),interval=myconfig.getint('default', 'interval'), outfile_stem=outfile+"_A")
        logging.info(candidates[:10])
        surprising = [(cand, score) for (cand, score) in candidates if score > 0.9]
        logging.info(len(surprising))
        with open(outfile+"_A", 'w') as outstream:
            for (term, score) in candidates:
                if score < 0.505:
                    break
                else:
                    outstream.write("{}\t{}\n".format(term, score))


        candidates = bootstrap_compare(samB, samA, field=field, repeats=myconfig.getint('default', 'repeats'),
                                       prop=myconfig.getint('default', 'prop'),
                                       interval=myconfig.getint('default', 'interval'), outfile_stem=outfile+"_B")
        logging.info(candidates[:10])
        surprising = [(cand, score) for (cand, score) in candidates if score > 0.9]
        logging.info(len(surprising))
        with open(outfile+"_B", 'w') as outstream:
            csvwriter=csv.writer(outstream,delimiter='\t',quotechar='\"',quoting=csv.QUOTE_MINIMAL)
            for (term, score) in candidates:
                if score < 0.505:
                    break
                else:
                    csvwriter.writerow([term,score])

