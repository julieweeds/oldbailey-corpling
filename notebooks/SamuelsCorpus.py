import pandas as pd
import numpy as np
import matplotlib as plt
from collections import defaultdict
import random
import spacy,operator,math,json
import logging
import configparser,ast,os,sys
from time import time
import CharacterisingFunctions as cf

from spacy.tokens import Doc


class ListTokenizer(object):
    #pass tokens in a list to Spacy
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, tokenlist):
        words = tokenlist
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


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

def make_index(columns):
    cindex={}
    for i,col in enumerate(columns):
        cindex[col]=i+1
    return cindex


class SamuelsCorpus:
    alias = {'q#TOTEN': 'vard', '#TOTEN': 'vard'}
    semtag_split = {'SEMTAG1': ' ', 'SEMTAG2': ';', 'SEMTAG3': ';'}

    def __init__(self):
        #this will be overrided in general

        self.rows=[]
        self.cols=[]

    def get_dataframe(self):

        try:
            return self.dataframe
        except:
            self.dataframe = pd.DataFrame.from_records(self.rows, columns=self.cols)
            return self.dataframe

class Processor(SamuelsCorpus):
    #functionality for processing corpus and outputting files

    def __init__(self, filepaths, parentdir='',outfile='',chunksize=1, lowercase=True, nlp=None, semtag_first=True):
        self.filenames=filepaths
        self.filepaths = [os.path.join(parentdir,f) for f in self.filenames]

        if outfile=='':
            self.outfile=self.filenames[0]+"_combined.csv"
            self.ppmimatfile=self.filenames[0]+"_cooccurrence.json"
        else:
            self.outfile=outfile+"_combined.csv"
            self.ppmimatfile=outfile+"_coocccurrence.json"

        self.chunksize = chunksize
        self.lowercase = lowercase
        self.semtag_first = semtag_first

        self.rows = []
        self.cols = ['fileid', 'chunk', 'sentence', 'id']
        self.spacy_columns = ['chunk', 'sentence', 'id', 'token', 'pos', 'lemma', 'entity_type', 'gram_dep',
                              'gram_head']
        if self.lowercase:
            self.cols.append('vard_lower')

        if nlp == None:
            print("Initialising Spacy")
            self.nlp = spacy.load('en', create_make_doc=ListTokenizer)
        else:
            self.nlp = nlp
        self.loadfile()

    def reverse_header(self, header):
        mydict = {}
        for i, text in enumerate(header):
            mydict[i] = SamuelsCorpus.alias.get(text, text)
        return mydict

    def addtocols(self, values):
        for value in values:
            if value not in self.cols:
                self.cols.append(value)

    def update_semtags(self, row):
        newrow = {}
        for key in row.keys():
            if key in SamuelsCorpus.semtag_split.keys():
                splitchar = SamuelsCorpus.semtag_split[key]
                field = str(row[key])
                parts = field.split(splitchar)
                newrow[key] = parts[0]
            else:
                newrow[key] = row[key]
        return newrow

    def loadfile(self):
        self.errors = []
        for filepath in self.filepaths:
            print("Reading {}".format(filepath))
            with open(filepath) as instream:
                newfile = True
                self.sentences = -1
                insent = 0
                self.allsentences = -1
                oldchunk = 0
                for count, line in enumerate(instream):
                    row = {}
                    line = line.rstrip()
                    parts = line.split('\t')
                    if newfile:
                        headerdict = self.reverse_header(parts)
                        self.addtocols(headerdict.values())
                        newfile = False
                    elif len(parts) == 7:  # well-formed line
                        insent += 1
                        for i, field in enumerate(parts):
                            row[headerdict[i]] = field
                        # print(row)

                        if row['vard'] == 'S_BEGIN':
                            self.sentences += 1
                            self.allsentences += 1
                            insent = insent - 2
                        row['chunk'] = int((self.allsentences) / self.chunksize)
                        if row['chunk'] != oldchunk:
                            # new chunk
                            self.sentences = 0
                            insent = -1

                        if self.sentences >= self.chunksize:
                            print(self.sentences, self.allsentences, self.chunksize, row['chunk'], oldchunk)
                        oldchunk = row['chunk']

                        if self.lowercase:
                            row['vard_lower'] = row['vard'].lower()

                        row['id'] = insent
                        row['fileid'] = count
                        row['sentence'] = self.sentences
                        # row['key']="{}:{}:{}".format(row['chunk'],row['sentence'],row['insent'])
                        if self.semtag_first:
                            row = self.update_semtags(row)
                        self.rows.append(row)
                    else:
                        self.errors.append(line)

        print("Read {}, errors = {}".format(count, len(self.errors)))
        self.chunks = int(self.allsentences / self.chunksize) + 1
        print("{} chunks of sentences".format(self.chunks))



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

    def make_tokenised_docs(self, field='vard', reset=False):
        if reset:
            try:
                del self.tokenised_docs
            except:
                pass
        try:
            return self.tokenised_docs
        except:
            docs = []
            docbuffer = 0
            thisdoc = []
            for i, line in enumerate(self.rows):
                currdoc = line['chunk']
                if currdoc != docbuffer:
                    docs.append(thisdoc)
                    thisdoc = []
                    docbuffer = currdoc
                if do_include_text(line):
                    thisdoc.append(line[field])
            docs.append(thisdoc)
            self.tokenised_docs = docs
            return docs

    def run_spacy(self, field='vard', reset=False):

        if reset:
            try:
                del self.nlpdocs
            except:
                pass

        try:
            return self.nlpdocs
        except:
            print("Extracting tokens")
            docs = self.make_tokenised_docs(field=field, reset=reset)
            print("Running spacy")
            self.nlpdocs = [self.nlp(doc) for doc in docs]
            return self.nlpdocs

    def get_spacy_frame(self, field='vard', reset=False):
        if reset:
            try:
                del self.spacy_frame
            except:
                pass
        try:
            return self.spacy_frame
        except:
            nlpdocs = self.run_spacy(field=field, reset=reset)
            spacy_rows = []
            for c, nlpdoc in enumerate(nlpdocs):

                for s, sent in enumerate(nlpdoc.sents):
                    if s >= self.chunksize:
                        print("Warning: Ignoring new sentence {},{}".format(c, s))
                        s = self.chunksize - 1
                    for token in sent:
                        row = {}
                        row['chunk'] = c
                        row['sentence'] = s
                        row['id'] = token.i
                        row['token'] = token.text
                        row['pos'] = token.pos_
                        row['lemma'] = token.lemma_
                        row['entity_type'] = token.ent_type_
                        row['gram_dep'] = token.dep_
                        row['gram_head'] = token.head.i
                        # row['key']="{}:{}:{}".format(row['chunk'],row['sentence'],row['id'])
                        spacy_rows.append(row)

            self.spacy_frame = pd.DataFrame.from_records(spacy_rows, columns=self.spacy_columns)
            return self.spacy_frame

    def get_combined(self, field='vard', reset=False):
        df = self.get_dataframe()
        sdf = self.get_spacy_frame(field=field, reset=reset)
        df = df[df['LEMMA'] != 'NULL']
        cdf = pd.merge(df, sdf, on=['id', 'chunk', 'sentence'])
        return cdf

    def extract_row(self, row, field='vard'):


        start = row[self.columnindex[field]]
        rel = row[self.columnindex['gram_dep']]
        head = row[self.columnindex['gram_head']]
        chunk = row[self.columnindex['chunk']]
        feat = self.cdf[self.cdf['chunk'] == chunk][self.cdf['id'] == head][field]
        for thing in feat:
            end = thing
        # print("{}:{}:{}".format(start,rel,end))
        try:
            forward = "{}:{}".format(rel, end)
            backward = "_{}:{}".format(rel, start)
        except:
            print("Error processing feature in chunk {}".format(chunk))
            print(row)
            print(head, field, feat)
            return
        adict = self.features.get(start, None)
        if adict == None:
            adict = {}
        adict[forward] = adict.get(forward, 0) + 1
        self.features[start] = adict
        self.rowtotals[start] = self.rowtotals.get(start, 0) + 1
        self.columntotals[forward] = self.columntotals.get(forward, 0) + 1
        self.grandtotal += 1
        bdict = self.features.get(end, None)
        if bdict == None:
            bdict = {}
        bdict[backward] = bdict.get(backward, 0) + 1
        self.features[end] = bdict
        self.rowtotals[end] = self.rowtotals.get(end, 0) + 1
        self.columntotals[backward] = self.columntotals.get(backward, 0) + 1
        self.grandtotal += 1

    def extract(self, field='vard', reset=False):
        # keys=self.df['id'].unique()
        # print(keys)
        if reset:
            try:
                del self.features
                del self.rowtotals
                del self.columntotals
                del self.grandtotal
            except:
                pass

        try:
            return self.features
        except:
            print("Extracting features")
            self.features = {}
            self.rowtotals = {}
            self.columntotals = {}
            self.grandtotal = 0

            count = 0
            for row in self.cdf.itertuples():
                self.extract_row(row, field=field)
                count += 1
                if count % 10000 == 0:
                    print("Processed {} rows".format(count))

        return self.features

    def convert_to_ppmi(self, reset=False):

        if reset:
            try:
                del self.pmi_matrix
            except:
                pass

        try:
            return self.pmi_matrix
        except:

            self.pmi_matrix = {}
            for key, featdict in self.features.items():
                pmi_feats = {}
                for feat, value in featdict.items():
                    rowtotal = self.rowtotals[key]
                    coltotal = self.columntotals[feat]
                    pmi = math.log((featdict[feat] * self.grandtotal) / (rowtotal * coltotal))

                    if pmi > 0:
                        pmi_feats[feat] = pmi
                self.pmi_matrix[key] = pmi_feats
            return self.pmi_matrix

    def run(self,field='SEMTAG3'):

        print("Adding Spacy annotations")
        self.cdf=self.get_combined(reset=True)
        self.cdf.to_csv(self.outfile,na_rep='NULL')
        self.columnindex=make_index(self.cdf.columns)

        print("Extracting dependency features")
        self.extract(field=field,reset=True)
        print("Converting to PPMI")
        ppmi_matrix=self.convert_to_ppmi(reset=True)
        with open(self.ppmimatfile,'w') as outstream:
            json.dump(ppmi_matrix,outstream)
        print("Completed successfully, writing {} and {}".format(self.outfile,self.ppmimatfile))


class Viewer(SamuelsCorpus):
    #functionality for viewing corpus and distributions

    def __init__(self,infile,parentdir='',lowercase=True,refdf=False):
        #need to set up and load in dataframes and ppmi_matrices
        self.lowercase=lowercase
        self.parentdir=parentdir


        if refdf:
            self.dataframe=infile
            self.columnindex=make_index(self.dataframe.columns)
        else:
            self.infile=infile
            self.loadfiles()


    def loadfiles(self):
        dffile=os.path.join(self.parentdir,self.infile+"_combined.csv")
        matfile=os.path.join(self.parentdir,self.infile+"_cooccurrence.json")

        self.dataframe = pd.read_csv(dffile,keep_default_na=False,na_values=['NaN'])
        self.columnindex = make_index(self.dataframe.columns)
        with open(matfile,'r') as instream:
            self.pmi_matrix=json.load(instream)

    def get_pmimatrix(self):
        try:
            return self.pmi_matrix
        except:
            print("Error: No pmi matrix defined for this viewer (it is probably a reference dataframe for corpora comparison)")

    def make_bow(self, field='vard', k=100000,cutoff=0,displaygraph=False):
        # turn corpus into a bag of words for a certain field - variant of make_hfw_dist()

        sumdict = {}
        corpussize = 0
        df = self.get_dataframe()
        df = df[df['LEMMA'] != 'NULL']
        if self.lowercase and field == 'vard':
            field = 'vard_lower'
        for item in df[field]:
            sumdict[item] = sumdict.get(item, 0) + 1
            corpussize += 1

        print("Size of corpus is {}".format(corpussize))
        candidates = sorted(sumdict.items(), key=operator.itemgetter(1), reverse=True)
        if cutoff>0:
            for cand,score in candidates[:cutoff]:
                print("({},{}) : {}".format(cand,score,self.find_text(cand,field=field)))
                #print("{}:{}".format(cand,score))

        if displaygraph:
            cf.display_list([(corpussize,candidates[:k])],cutoff=cutoff,xlabel=field+' (High Frequency)')
        return corpussize, candidates[:k]

    def find_text(self, semtag, field='SEMTAG3'):

        # return hf word distribution for given tag
        #print(semtag)
        semtag = self.match_tag(semtag, field=field)
        df = self.get_dataframe()
        df = df[df['LEMMA'] != 'NULL']
        if self.lowercase:
            groupby = 'vard_lower'
        else:
            groupby = 'vard'
        mylemmas = df[df[field] == semtag].groupby(groupby)['fileid'].nunique()
        mylemmas = mylemmas.sort_values(ascending=False)
        # print(mylemmas)
        mylist = list(mylemmas[0:10].index.values)
        mylist = [(t, mylemmas[t]) for t in mylist]
        return mylist

    def match_tag(self, brief, field='SEMTAG3'):

        # find the most likely matching tag for the string in brief (intended to match ZF to ZF [Pronoun] etc.)
        req = 5  # number of matches to be found as ratio e.g. 5:1
        matches = {}
        tagset = self.get_dataframe()[field]
        tagset=tagset.dropna()
        candidates = [(brief, 1)]
        for tag in tagset:
            #print(tag)
            parts = tag.split(' ')
            if brief == parts[0] or brief == tag:
                matches[tag] = matches.get(tag, 0) + 1
                candidates = sorted(matches.items(), key=operator.itemgetter(1), reverse=True)
                if len(candidates) > 1:
                    ratio = candidates[0][1] / candidates[1][1]
                    if ratio > req:
                        break
                elif candidates[0][1] > req:
                    break
        return candidates[0][0]

    def find_tags(self, word, field='SEMTAG3'):
        # find the tags given to a given word

        if self.lowercase:
            sourcefield = 'vard_lower'
        else:
            sourcefield = 'vard'

        df = self.get_dataframe()
        mytags = df[df[sourcefield] == word].groupby(field)['fileid'].nunique()
        mytags = mytags.sort_values(ascending=False)
        mylist = list(mytags[0:10].index.values)
        mylist = [(t, mytags[t]) for t in mylist]
        return mylist

    def get_one_best_features(self, key, cutoff=10,field='SEMTAG3'):

        key=self.match_tag(key,field=field)
        featdict = self.pmi_matrix[key]
        candidates = sorted(featdict.items(), key=operator.itemgetter(1), reverse=True)
        return candidates[0:cutoff]


    def get_best_features(self, cutoff=10):
        pmi_matrix=self.get_pmimatrix()

        best = {}
        for key in pmi_matrix.keys():
            best[key] = self.get_one_best_features(self, key, cutoff=cutoff)

        return best


class Comparator:

    def __init__(self,filedict,parentdir=''):

        self.filedict=filedict
        self.parentdir=parentdir

        self.viewerdict=self.init_viewers()


    def init_viewers(self):

        viewerdict={}
        for key in self.filedict.keys():
            viewerdict[key]=Viewer(self.filedict[key])
        return viewerdict

    def get_reference_viewer(self):

        keys=list(self.viewerdict.keys())

        df=self.viewerdict[keys[0]].get_dataframe()

        for key in keys[1:]:
            df=df.append(self.viewerdict[key].get_dataframe())

        return Viewer(df,refdf=True)

    def compute_surprises(self,key,field='SEMTAG3',measure='llr',cutoff=0,displaygraph=False):
        tagbag=self.viewerdict[key].make_bow(field=field)
        reftagbag=self.get_reference_viewer().make_bow(field=field)

        distinctive_tags=cf.improved_compute_surprises(tagbag,reftagbag,measure=measure,k=cutoff,display=False)
        if cutoff>0:
            print("Number of characteristic tags is {}".format(len(distinctive_tags)))
            for(tag,score) in distinctive_tags[:cutoff]:
                print("({}, {}) : {}".format(tag,score,self.viewerdict[key].find_text(tag,field=field)))

        if displaygraph and len(distinctive_tags)>0:
            cf.display_list([(-1,distinctive_tags)],cutoff=cutoff,xlabel=field+' (Characteristic)',ylabel=measure+" (Score)")

        return distinctive_tags