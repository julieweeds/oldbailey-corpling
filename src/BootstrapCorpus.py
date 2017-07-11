import pandas as pd
import matplotlib as plt
from collections import defaultdict
import random
import nlp_tools
import spacy,operator
import logging
import configparser,ast,os,sys


def make_countdict(alldata):
    countdict = {}
    blacklist = ['words', 'obc_hiscoCode']

    for heading in alldata.columns:
        # print('Generating counts for ' +heading)
        if heading not in blacklist:
            countdict[heading] = defaultdict(int)
            selection = alldata[heading]
            for item in selection:
                # print(item)
                countdict[heading][item] += 1
        else:
            # print('skipping')
            pass

    return countdict


def validated(reqlist, valuedata):
    reqdict = {}
    for (field, value) in reqlist:

        parts = field.split(':')
        if len(parts) == 1:
            if field in valuedata.keys():
                if isinstance(value, list):
                    ok = []
                    for v in value:
                        if v in valuedata[field].keys():
                            ok.append(v)
                    if len(ok) > 0:
                        reqdict[field] = ok
                elif value in valuedata[field].keys():
                    reqdict[field] = value


        else:
            if (parts[1] == "max" or parts[1] == "min") and parts[0] in valuedata.keys():

                if isinstance(value, list):
                    logging.warning("Error: min and max cannot be list")

                elif value in valuedata[parts[0]].keys() and isinstance(value, int):
                    reqdict[field] = value

    return reqdict


def find_trials(worddf, trialdf,reqlist, join='obo_trial'):
    trialreqdict = validated(reqlist, make_countdict(trialdf))
    wordsreqdict = validated(reqlist, make_countdict(worddf))

    logging.info(trialreqdict)
    logging.info(wordsreqdict)
    ok = True
    for (req, _value) in reqlist:
        if req in trialreqdict.keys() or req in wordsreqdict.keys():
            pass
        else:
            logging.warning("Requirement {} not satisfied".format(req))
            ok = False

    if not ok:
        return None

    trials = trialdf
    for req in trialreqdict.keys():
        parts = req.split(':')
        value = trialreqdict[req]
        if len(parts)>1:
            if parts[1]=='max':
                trials=trials[trials[parts[0]]<=value]
            elif parts[1]=='min':
                trials=trials[trials[parts[0]]>=value]
        elif isinstance(value,list):
            trials=trials[trials[req].isin(value)]
        else:
            trials=trials[trials[req]==value]


    selection=[line for line in trials[join]]
    return selection


def bootstrap1(wdf, tdf, reqs):
    trials = find_trials(wdf, tdf, reqs)
    # print(len(trials),trials)
    c = bootstrap_corpus(wdf, trials, reqs)
    print(c)


def bootstrap_corpus(worddata, trials, reqs,prop=100):
    N = len(trials)
    corpus = []
    N=int(N*prop/100)
    allreqdict = validated(reqs, make_countdict(worddata))
    for i in range(0, N):
        atrial = random.choice(trials)
        wdf = worddata[worddata['obo_trial'] == atrial]
        for req in allreqdict.keys():
            parts = req.split(':')
            value = allreqdict[req]
            if len(parts) > 1:
                if parts[1] == 'max':
                    wdf = wdf[wdf[parts[0]] <= value]
                elif parts[1] == 'min':
                    wdf = wdf[wdf[parts[0]] >= value]
            elif isinstance(value, list):
                wdf = wdf[wdf[req].isin(value)]
            else:
                wdf = wdf[wdf[req] == value]

        corpus += [line for line in wdf['words']]
    return corpus


# For a given set of corpora, find the frequency distribution of the k highest frequency words
# Output total size of corpus and sorted list of term, frequency pairs

def find_hfw_dist(corpora, k=100000):
    # add worddicts for individual corpora
    # sort and output highest frequency words
    # visualise

    sumdict = {}
    corpussize = 0
    for acorpus in corpora:
        for (key, value) in acorpus.allworddict.items():
            sumdict[key.lower()] = sumdict.get(key.lower(), 0) + value
            corpussize += value

    logging.info("Size of corpus is {}".format(corpussize))
    candidates = sorted(sumdict.items(), key=operator.itemgetter(1), reverse=True)
    # print(candidates[:50])
    # print(len(sumdict))
    # print(sumdict)
    return corpussize, candidates[:k]


def compare(corpusA, corpusB, indicatordict):
    sizeA, hfwA = find_hfw_dist([corpusA])
    sizeB = corpusB.wordtotal

    for (word, freqA) in hfwA:
        freqB = corpusB.allworddict.get(word, 0)
        probA = freqA / sizeA
        probB = freqB / sizeB
        if probA > probB:
            indicatordict[word] = indicatordict.get(word, 0) + 1
    return indicatordict


def bootstrap_compare(corpusAreqs, allreqs, worddata, trialdata, repeats=10, prop=100):
    logging.info("Finding trials to meet requirements")
    trialsB = find_trials(worddata, trialdata, allreqs)
    logging.info(len(trialsB))
    trialsA = find_trials(worddata, trialdata, allreqs + corpusAreqs)
    logging.info(len(trialsA))
    indicatordict = {}
    for i in range(0, repeats):
        logging.info("Bootstrapping corpusB repetition {}".format(i))
        corpB = bootstrap_corpus(worddata, trialsB, allreqs,prop=prop)
        logging.info("Analysing corpus")
        corpusB = nlp_tools.corpus(corpB, nlp, prop=100, ner=False, loadfiles=False)
        for j in range(0, repeats):
            logging.info("Bootstrapping corpusA repetition {}".format(j))
            corpA = bootstrap_corpus(worddata, trialsA, allreqs + corpusAreqs,prop=prop)
            logging.info("Analysing corpus")
            corpusA = nlp_tools.corpus(corpA, nlp, prop=100, ner=False, loadfiles=False)
            logging.info("Comparing corpora")
            indicatordict = compare(corpusA, corpusB, indicatordict)

    logging.info("Generating candidates")
    N = repeats * repeats
    candidates = [(term, (value + 1) / (N + 1)) for (term, value) in indicatordict.items()]
    sortedlist = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    return sortedlist


if __name__=="__main__":


    nlp=spacy.load('en')


    myconfig=configparser.ConfigParser()
    if len(sys.argv)>1:
        configfile=sys.argv[1]
    else:
        configfile='bootstrap.cfg'
    myconfig.read(configfile)

    logging.basicConfig(level=logging.INFO)
    parentdir=myconfig.get('default','parentdir')
    worddata = pd.DataFrame.from_csv(os.path.join(parentdir,myconfig.get('default','worddatafile')), sep='\t')
    trialdata = pd.DataFrame.from_csv(os.path.join(parentdir,myconfig.get('default','trialdatafile')), sep='\t')

    outfiles=ast.literal_eval(myconfig.get('default','outfile'))

    allreqlist = ast.literal_eval(myconfig.get('default','allreqlist'))
    Areqs=ast.literal_eval(myconfig.get('default','Areqs'))

    for Areq,outfile in zip(Areqs,outfiles):

        candidates=bootstrap_compare(Areq,allreqlist,worddata,trialdata,repeats=myconfig.getint('default','repeats'),prop=myconfig.getint('default','prop'))
        logging.info(candidates[:10])
        surprising=[(cand,score) for (cand,score) in candidates if score > 0.9]
        logging.info(len(surprising))
        with open(outfile,'w') as outstream:
            for (term,score) in candidates:
                outstream.write("{}\t{}\n".format(term,score))

