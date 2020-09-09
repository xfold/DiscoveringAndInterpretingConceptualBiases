from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from operator import itemgetter
from scipy import spatial
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
import inflect
import numpy as np
import statistics
import json
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime


from importlib import reload
from Misc import USAS_utils
reload(USAS_utils)

'''
This class performs the complete analysis of bias, given an embedding model and a set of target sets

'''
class DADDBias:
    def __init__(self, modelpath=None,  outputfolder=None, name=None):
        '''
        modelpath <word2vec model> : path to the embedding model        
        outputfolder <str> : Output folder path
        '''
        #init vars
        self.model = None
        if(modelpath is not None):
            self.model = Word2Vec.load(modelpath)
        self.outputfolder = outputfolder
        self.name = name
        if(self.name is None and self.model is not None):
            self.name = modelpath[min(-30, len(modelpath)):]
        
        #bias
        self.performed_bias = False
        self.acceptedPOS = None
        self.stdevs = None  #standard deviation min threshold
        self.sid = SentimentIntensityAnalyzer()
        self.tset1 = None   #target set 1
        self.tset2 = None   #target set 2
        self.tset1_centroid = None # centroids of target sets
        self.tset2_centroid = None # centroids of target sets
        self.stdevs1_thr = None #salience threshold for target set 1
        self.stdevs2_thr = None #salience threshold for target set 2
        self.b1_dict = None  #dictionary of biased words towards target set 1
        self.b2_dict = None  #dictionary of biased words towards target set 2
        
        #cluster
        self.performed_cluster = False
        self.kstart = None
        self.kend = None
        self.krepetitions = None
        self.clusters1 = None
        self.clusters2 = None
        self.clusters1_dict = None #cluster dicitonary, in which key are all word in the cluster separated by space and value are satistics about the cluster
        self.clusters2_dict = None
        self.cluster1_silhouette = None
        self.cluster2_silhouette = None
        
        #USAS
        self.performed_USAS = False
        self.usasDict1 = None
        self.usasDict2 = None
        self.usasLabelsRanking1 = None
        self.usasLabelsRanking2 = None

    def Save(self, forcepath = None):
        '''
        Save the actual __dict__ elements in a json, ignoring model and sentiment analysed
        '''
        if(forcepath is not None):
            filename = "{}/{}".format(self.outputfolder,forcepath)
        else:
            filename = "{}/{}_bias.{}_cluster.{}_USAS.{}.json".format(self.outputfolder,
                                                                 self.name,
                                                                 self.performed_bias,
                                                                 self.performed_cluster,
                                                                 self.performed_USAS)
            
        #save self, excluding the embedding model to reduce size
        del self.model
        del self.sid
        js = json.dumps(self.__dict__)
        fp = open(filename, 'w+')
        fp.write(js)
        fp.close()
        return filename
    
    def Load(self, jsonpath):
        '''
        Load the actual __dict__ elements from a json, which does not include model not sentiment analiser
        '''
        with open(jsonpath) as f:
            x = json.load(f)
            self.__dict__ = x
            self.sid = SentimentIntensityAnalyzer()
            self.model = None
        
        
    def _calculateCentroid(self, wordlist):
        '''
        Calculate centroid of the wordlist list of words based on the model embedding vectors
        '''
        centr = np.zeros( len(self.model.wv[wordlist[0]]) )
        for w in wordlist:
            centr += np.array(self.model.wv[w])
        return centr/len(wordlist)
    
    def _keepOnlyModelWords(self, words):
        aux = [ word for word in words if word in self.model.wv.vocab.keys()]
        return aux
    
    def _getWordFreq(self, word):
        if word in self.model.wv.vocab:
            wm = self.model.wv.vocab[word]
            return [word, wm.count, wm.index]
        return None
    
    def _getModelMinMaxRank(self):
        minF = 999999
        maxF = -1
        for w in self.model.wv.vocab:
            wm = self.model.wv.vocab[w] #wm.count, wm.index
            rank = wm.index
            if(minF>rank):
                minF = rank
            if(maxF<rank):
                maxF = rank
        return [minF, maxF]
        
    def _getSentiment(self,word):
        return self.sid.polarity_scores(word)['compound']
    
    '''
    Normalises a value in the positive space
    '''    
    def _normalise(self, val, minF, maxF):
        #print(val, minF, maxF)
        if(maxF<0 or minF<0 or val<0):
            raise Exception('All values should be in the positive space. minf: {}, max: {}, freq: {}'.format(minF, maxF, val))
        if(maxF<= minF):
            raise Exception('Maximum frequency should be bigger than min frequency. minf: {}, max: {}, freq: {}'.format(minF, maxF, freq))
        val -= minF
        val = val/(maxF-minF)
        return val
        
    def _getCosineDistance(self, wv1, wv2):
        return spatial.distance.cosine(wv1, wv2)
    
    def _getMinMax(self, dict_value):
        l = list(dict_value.values())
        return [ min(l), max(l)]
    
    def _findStDevThresholdSal(self, dwords, stdevs):
        '''
        dword is an object like {'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'freqW':freqW, 'sal':val, 'wv':wv, 'sent':sent }
        stdevs : minimum stdevs fow hich we want to compute the threshold
        
        returns
        outlier_thr : the threhsold correpsonding to stdevs considering salience values from the dwrods object list
        '''
        allsal = []
        for obj in dwords:
            allsal.append(obj['sal'])
        stdev = statistics.stdev(allsal)
        outlier_thr = (stdev*stdevs)+sum(allsal)/len(allsal)
        return outlier_thr
        
    
    def CalculateBiasedWords(self, targetset1, targetset2, minstdev, 
                             acceptedPOS = ['JJ', 'JJS', 'JJR','NN', 'NNS', 'NNP', 'NNPS','VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], 
                             words = None, force=False):
        '''
        this function calculates the list of biased words towards targetset1 and taregset2 with slaience > than the 
        specified times (minstdev) the standard deviation.
        
        targetset1 <list of strings> : target set 1
        targetset2 <list of strings> : target set 2
        minstdev int : Minium threhsold for stdev to select biased words
        acceptedPOS <list<str>> : accepted list of POS to consider for the anamlysis, as defined in NLTK POS tagging lib. 
                                  If None, no POS filtering is applied and all words in the vocab are considered
        words list<str> : list of words we want to consider. If None all words in the vocab are considered
        '''
        if(self.model is None):
            raise Exception("You need to define a model to estimate biased words.")
        if(targetset1 is None or targetset2 is None):
            raise Exception("Target sets are necessary to estimate biased words.")
        if(minstdev is None):
            raise Exception("You need to define a minimum threshold for standard deviation to select biased words.")
        if(self.performed_bias and not force):
            raise Exception("The bias execution has alreayd been performed in this model. To repeat it, set force parameter to true")
        
        self.acceptedPOS = acceptedPOS
        self.tset1 = self._keepOnlyModelWords(targetset1)
        self.tset2 = self._keepOnlyModelWords(targetset2)
        self.stdevs = minstdev
        
        #we remove words in the target sets, and also their plurals from the set of interesting words to process.
        engine = inflect.engine()
        toremove = targetset1 + targetset2 + [engine.plural(w) for w in targetset1] + [engine.plural(w) for w in targetset2]
        if(words is None):
            words = [w for w in self.model.wv.vocab.keys() if w not in toremove]
        
        #calculate centroids 
        self.tset1_centroid = self._calculateCentroid(self.tset1)
        self.tset2_centroid = self._calculateCentroid(self.tset2)
        [minR, maxR] = self._getModelMinMaxRank()
        
        #get biases for words
        biasWF = {}
        biasWM = {}
        for i, w in enumerate(words):
            p = nltk.pos_tag([w])[0][1]
            if acceptedPOS is not None and p not in acceptedPOS:
                continue
            wv = self.model.wv[w]
            diff = self._getCosineDistance(self.tset2_centroid, wv) - self._getCosineDistance(self.tset1_centroid, wv)
            if(diff>0):
                biasWF[w] = diff
            else:
                biasWM[w] = -1*diff

        #get min and max bias for both target sets, so we can normalise these values later
        [minbf, maxbf] = self._getMinMax(biasWF)
        [minbm, maxbm] = self._getMinMax(biasWM)
        
        #iterate through all 'selected' words
        biased1 = []
        biased2 = []
        for i, w in enumerate(words):
            #print('..Processing ', w)
            p = nltk.pos_tag([w])[0][1]
            if acceptedPOS is not None and p not in acceptedPOS:
                continue
            wv = self.model.wv[w]
            #sentiment
            sent = self._getSentiment(w)
            #rank and rank norm
            freq = self._getWordFreq(w)[1]
            rank = self._getWordFreq(w)[2]
            rankW = 1-self._normalise(rank, minR, maxR) 
            
            #normalise bias
            if(w in biasWF):
                bias = biasWF[w]
                biasW = self._normalise(bias, minbf, maxbf)
                val = biasW * rankW
                biased1.append({'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'rank':rank, 'rankW':rankW, 'sal':val, 'wv':wv.tolist(), 'sent':sent } ) 
            if(w in biasWM):
                bias = biasWM[w]
                biasW = self._normalise(bias, minbm, maxbm)
                val = biasW * rankW
                biased2.append({'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'rank':rank, 'rankW':rankW, 'sal':val, 'wv':wv.tolist(), 'sent':sent } ) 
        
        #calculate the salience threshold for both word sets, and select the list of biased words
        self.stdevs1_thr = self._findStDevThresholdSal(biased1, self.stdevs)
        self.stdevs2_thr = self._findStDevThresholdSal(biased2, self.stdevs)
        biased1.sort(key=lambda x: x['sal'], reverse=True)
        self.b1_dict = {}
        for k in biased1:
            if(k['sal']>=self.stdevs1_thr):
                self.b1_dict[k['word']] = k
        biased2.sort(key=lambda x: x['sal'], reverse=True)
        self.b2_dict = {}
        for k in biased2:
            if(k['sal']>=self.stdevs2_thr):
                self.b2_dict[k['word']] = k
        
        #transform centroid tol list so they become serializable
        self.tset1_centroid = self.tset1_centroid.tolist() 
        self.tset2_centroid = self.tset2_centroid.tolist()
        self.performed_bias = True
        return [self.b1_dict, self.b2_dict]
    
    def Clustering(self, repeateachclustering = 100, forcekmin=None, forcekmax=None, kjump=1, force = False):
        '''
        This function clusters similar words in concepts. It selects the best partition by means of silhouette cluster value, 
        or by forcing specific k-values for the k-means clustering.
        Before running this function, the set of most biased words need to be sleected by calling CalculateBiasedWords.
        
        repeateachclustering int : Number of times we want to repeat the partitoins for each k to find the best silhouette posisble
        forcekmin int : minimum k for kmeans to start exploring 
        forcekmax int : maximum k for kmeans to end exploring
        '''
        if(self.performed_bias is None or self.performed_bias == False):
            raise Exception("You first need to run CalculateBiasedWords before clustering.")
        if(self.b1_dict is None or len(self.b1_dict) == 0 or self.b2_dict is None or len(self.b2_dict) == 0):
            raise Exception("You first need to run CalculateBiasedWords before clustering.")
        if(repeateachclustering is None or repeateachclustering < 0):
            raise Exception("The repeateachclustering needs to be set, and needs to be a positive integer.")
        if(forcekmin is not None and forcekmax is not None and forcekmin >= forcekmax ):
            raise Exception("forcekmin needs to be smaller than forcekmax.")
        if(self.performed_cluster and not force):
            raise Exception("The clustering has already been performed in this model. To repeat it, set force parameter to true")
            
        self.kstart = 0
        if(forcekmin is not None):
            self.kstart = forcekmin
        self.kend = 200
        if(forcekmax is not None):
            self.kend = forcekmax
        self.krepetitions = repeateachclustering
        
        #prepare list of words and wv for both targetsets
        l1 = ([],[])
        for item in self.b1_dict.values():
            l1[0].append( item['word'] )
            l1[1].append( item['wv'] )
        l2 = ([],[])
        for item in self.b2_dict.values():
            l2[0].append( item['word'] )
            l2[1].append( item['wv'] )
        ll = (l1,l2)    
        
        for ts, l in enumerate(ll):
            words = l[0]
            wv = l[1]
            maxscore = [-1, None, None]
            for k in range(self.kstart, self.kend, kjump): 
                for repeat in range(0, self.krepetitions):  
                    clusterer = KMeans (n_clusters=k)
                    preds = clusterer.fit_predict(wv)
                    centers = clusterer.cluster_centers_
                    score = silhouette_score (wv, preds, metric='euclidean')
                    if(score>maxscore[0]):
                        maxscore[0] = score
                        maxscore[1] = preds
                        maxscore[2] = centers
                print('Exploring ',k, ' clusters... last silhouette score: ', score)        
            #we map the words to each cluster for the paritition with max silhouette score
            clusters = []
            for i in range(0,len(maxscore[2])):
                cl = []
                indexes = np.where(maxscore[1] == i)[0]
                for idx in indexes:
                    cl.append(words[idx])
                clusters.append(cl)
                
            #save the clusters
            if(ts == 0):
                self.clusters1 = clusters
                self.cluster1_silhouette = maxscore[0]
            else:
                self.clusters2 = clusters
                self.cluster2_silhouette = maxscore[0]
        
        
        #create cluster dictioanries
        self.clusters1_dict = self._createClusterDict(self.clusters1, self.b1_dict)
        self.clusters2_dict = self._createClusterDict(self.clusters2, self.b2_dict)
            
        self.performed_cluster = True
        return [self.clusters1, self.clusters2]
    
        
    def _createClusterDict(self, clusters, wdict):
        '''
        clusters <list<str>> : list of words in each cluster
        wdict : dictoinary of words, created in calculate biased words
        '''
        #{'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'rank':rank, 'rankW':rankW, 'sal':val,'wv':wv.tolist(), 'sent':sent
        d = {}
        for cluster in clusters:
            data = [0,0,0,0,0, np.zeros( len(list(self.b1_dict.values())[0]['wv']) )]  #biasW, freq, sent, sal,rankW, centroid
            key = " ".join(cluster)
            for w in cluster:
                data[0] += wdict[w]['biasW']
                data[1] += wdict[w]['freq']
                data[2] += wdict[w]['sent']
                data[3] += wdict[w]['sal']
                data[4] += wdict[w]['rankW']
                data[5] += np.array(wdict[w]['wv'])
            data[0]  = data[0] / len(cluster)    
            data[2]  = data[2] / len(cluster)    
            data[3]  = data[3] / len(cluster)    
            data[4]  = data[4] / len(cluster)    
            data[5]  = data[5] / len(cluster)    
            data[5] = data[5].tolist()
            d[key] = {'biasW_avg': data[0], 'freq':data[1], 'sent_avg': data[2], 'sal_avg':data[3], 'rankW_avg':data[4], 'centroid':data[5]}
        return d

    
    def _rankUSASLabels(self, cluster_dict):
        '''
        This method loads a set of clusters and counts how many times the USAS labels appear in the set of clusters
        returns
            list<string, int>, counting all USAS labels and frequency
        '''
        labeldict = {}
        labels = [v['USAS_label'].split(",") for v in cluster_dict.values()]
        #flatten the list of labels
        labels = list(itertools.chain.from_iterable(labels))
        labels = [l.strip() for l in labels]
        for l in labels:
            if(l in labeldict):
                labeldict[l] += 1
            else:
                labeldict[l] = 1
        tor = [(k,v) for k,v in labeldict.items()]
        tor.sort(key=lambda x: x[1], reverse=-1)
        return tor
        
    
    def USASLabels(self, force=False):
        '''
        Returns the most frequent label for all clusters, in the form of a dictionary where the key are the cluster
        elements separated by empty spaces and the value is the most frequent label or labels
        e.g. 
        self.usasDict1 = [{'word1 word2 word3' : 'label1,label2'},{...}]
        '''
        if(not self.performed_cluster or not self.performed_bias):
            raise Exception("you first need to perform the clustering to get the USAS labels")
        if(self.performed_USAS and not force):
            raise Exception("USAS has already been performed in this object. Set force to True to redo the USAS labelling")
        
        #get usas dicts
        self.usasDict1 = USAS_utils.getUsasDict(self.clusters1)
        self.usasDict2 = USAS_utils.getUsasDict(self.clusters2) 
        #update cluster info
        for k,v in self.usasDict1.items():
            self.clusters1_dict[k]['USAS_label'] = v
        for k,v in self.usasDict2.items():
            self.clusters2_dict[k]['USAS_label'] = v
        self.performed_USAS = True
        self.usasLabelsRanking1 = self._rankUSASLabels(self.clusters1_dict)
        self.usasLabelsRanking2 = self._rankUSASLabels(self.clusters2_dict)
        return [self.usasDict1, self.usasDict2]
    
    def _orderWordsByFrequency(self, wordL, wDict):
        '''
        wordl <list<str>> : list of words to order
        wDict biasdcit : the bias ict cotnainign information about the words to order
        '''
        wl = [ (w, wDict[w]['freq']) for w in wordL ]
        wl.sort(key=lambda x: x[1], reverse=True)
        return [w[0] for w in wl]
        
    
    def GetClusterRanking(self, rankingtype):
        '''
        this function orders the clusters based on a ranking method 'salience', 'frequency', 'sentiment', 'bias', 'rank'
        '''
        if(not self.performed_cluster or not self.performed_bias):
            raise Exception("you first need to perform the clustering to get cluster rankings")
        
        acceptedrankings = ['salience', 'frequency', 'sentiment', 'bias', 'rank', 'sentiment_pos']
        if(rankingtype not in acceptedrankings):
            raise Exception("the specified ranking type is not available, please select from {}".format(acceptedrankings))
        
        prop = ""
        reverse = True
        if(rankingtype == 'bias'):
            prop = 'biasW_avg'
            reverse = True
        if(rankingtype == 'rank'):
            prop = 'rankW_avg'
            reverse = True
        elif(rankingtype == 'sentiment'):
            prop = 'sent_avg'
            reverse = False
        elif(rankingtype == 'sentiment_pos'):
            prop = 'sent_avg'
            reverse = True
        elif(rankingtype == 'frequency'):
            prop = 'freq'
            reverse = True
        elif(rankingtype == 'salience'):
            prop = 'sal_avg'
            reverse = True
        
        c1list = [(self._orderWordsByFrequency( k.split(" "), self.b1_dict),v) for k,v in self.clusters1_dict.items()]
        c1list.sort(key=lambda x: x[1][prop], reverse=reverse)
        c2list = [(self._orderWordsByFrequency( k.split(" "), self.b2_dict),v) for k,v in self.clusters2_dict.items()]
        c2list.sort(key=lambda x: x[1][prop], reverse=reverse)
        
        return [c1list, c2list]
   

    def GetClustersWithUSASLabels(self, usaslabels, targetset=1):
        '''
        This function returns all clusters from the specified target set with the specified USAS label.
        usaslabel <list<str>> : list of usas labels we want to find the clusters
        targetset <int> 1|2 : target set we are interested in , only accepts values 1 or 2
        completelabel <bool> : Sepcifies whether we are only looking with complete labels or substrings
        returns:
        clusters matching the criteria
        '''
        if(not self.performed_cluster or not self.performed_bias or not self.performed_USAS):
            raise Exception("you first need to perform the USAS labelling to get the USAS labels")
        if(usaslabels is None):
            raise Exception("you need to send the USAS label as parameter")
        if(targetset is None or (targetset != 1 and targetset!=2)):
            raise Exception("Target set should be specified, and can only be 1 or 2")
        
        tor = []
        clusters = []
        if(targetset==1):
            clusters = [(k.split(" "),v) for k,v in self.clusters1_dict.items()]
        if(targetset==2):
            clusters = [(k.split(" "),v) for k,v in self.clusters2_dict.items()]
            
        for k in clusters:
            for label in usaslabels:
                if(label in k[1]['USAS_label']):
                    tor.append(k)
        return tor                
            
    def UpdateClusterDict(self):
        '''
        This functions adds an element in the cluster list to assign a usas label to each cluster
        '''
        if(not self.performed_cluster or not self.performed_bias or not self.performed_USAS):
            raise Exception("you first need to perform the USAS labelling to get the USAS labels")
        for k in self.clusters1_dict:
            usasl = self.usasDict1[k]
            self.clusters1_dict[k]['USAS_label'] = usasl
            
        for k in self.clusters2_dict:
            usasl = self.usasDict2[k]
            self.clusters2_dict[k]['USAS_label'] = usasl
      
            
    def GetClusterThatContainsWord(self, word):
        '''
        searches for word word in the set of clusters created
        '''
        for c in self.clusters1:
            if(word in c):
                clw = " ".join(c)
                return [c, self.clusters1_dict[clw]]
        for c in self.clusters2:
            if(word in c):
                clw = " ".join(c)
                return [c, self.clusters2_dict[clw]]
        return None
            
    def PrintSummary(self):
        print()
        print('---- {} ----'.format(self.name))
        print('> BIAS {} ----'.format(self.performed_bias))
        print('POS: {} '.format(self.acceptedPOS))
        print('s1size: {} '.format(len(self.b1_dict)))
        print('s2size: {} '.format(len(self.b2_dict)))
        
        print('> CLUSTER {} ----'.format(self.performed_cluster))
        print('tau: {} '.format(self.krepetitions))
        k1aux = [len(k) for k in self.clusters1]
        k2aux = [len(k) for k in self.clusters2]
        print('K1 size: {} '.format(len(self.clusters1)))
        print('   min: {}, max {}, avg {}, std dev {}'.format( min(k1aux), max(k1aux), np.mean(k1aux), statistics.stdev(k1aux) ))
        print('K2 size: {} '.format(len(self.clusters2)))
        print('   min: {}, max {}, avg {}, std dev {}'.format( min(k2aux), max(k2aux), np.mean(k2aux), statistics.stdev(k2aux) ))
        print('> USAS {} ----'.format(self.performed_USAS))
        
         
            
            
            
            
            
            
            
            
            
            

        
        
        
        
        