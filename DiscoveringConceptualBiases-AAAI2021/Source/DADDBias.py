'''
******************************************************************************************************************
NOTE: Although the code has been cleaned and commented, it is far from a final version. Many structures and procedures could be improved. 
We are still working on it, but hopefully this serves as an alpha version of the approach presented in the paper.
******************************************************************************************************************
'''


from gensim.models import Word2Vec
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
from Misc import USAS_utils


'''
This class performs the complete analysis of bias, given an embedding model and a set of attribute concepts (also named target sets).
This class contains all code used in the paper, including:
    1. Salient words  (using CalculateBiasedWords fnc)
    2. Cluster + silhouette  (using Clustering fnc)
    3. Semantic categorisation (using USASLabels fnc)
The object can be saved using Save() function, and later loaded using Load().
Figures can be created and visualised using FigureCreator.py
'''
class DADDBias:
    def __init__(self, modelpath=None,  outputfolder=None, name=None):
        '''
        modelpath <word2vec model> : path to the embedding model        
        outputfolder <str> : Output folder path
        name <str> : output name for the model, used when calling Save()
        '''
        #init vars
        self.model = None
        if(modelpath is not None):
            self.model = Word2Vec.load(modelpath)
        self.outputfolder = outputfolder
        self.name = name
        if(self.name is None and self.model is not None):
            self.name = modelpath[min(-30, len(modelpath)):]
        
        #bias variables
        self.performed_bias = False     #True/false check variable
        self.acceptedPOS = None         #accepted POS
        self.stdevs = None              #standard deviation min threshold
        self.tset1 = None               #target set 1 (attribute concept1)
        self.tset2 = None               #target set 2 (attribute concept2)
        self.tset1_centroid = None      #centroids of target sets
        self.tset2_centroid = None      #centroids of target sets
        self.stdevs1_thr = None         #salience threshold for target set 1
        self.stdevs2_thr = None         #salience threshold for target set 2
        self.b1_dict = None             #dictionary of biased words towards target set 1
        self.b2_dict = None             #dictionary of biased words towards target set 2
        self.sid = SentimentIntensityAnalyzer()
        
        #cluster variables
        self.performed_cluster = False  #True/false check variable
        self.kstart = None              #k strart value for kmeans cluster
        self.kend = None                #k ending value
        self.krepetitions = None        #tau repetitions of clustering
        self.clusters1 = None           #list of clusters from the selection partitoin for ts1
        self.clusters2 = None           #list of clusters from the selection partitoin for ts2
        self.clusters1_dict = None      #cluster dicitonary for ts1, in which key are all word in the cluster separated by space and value are satistics about the cluster
        self.clusters2_dict = None      #cluster dicitonary for ts2, in which key are all word in the cluster separated by space and value are satistics about the cluster
        self.cluster1_silhouette = None #silhouette value of selected partitoin for ts1 
        self.cluster2_silhouette = None #silhouette value of selected partitoin for ts2
        
        #semantic categorisation variables (USAS)
        self.performed_USAS = False     #True/false check variable
        self.usasDict1 = None           #dictionary for ts1, containing the selected usas label for each cluster
        self.usasDict2 = None           #dictionary for ts2, containing the selected usas label for each cluster
        self.usasLabelsRanking1 = None  #agrgegation of USAS labels at a partition lebel for ts1
        self.usasLabelsRanking2 = None  #agrgegation of USAS labels at a partition lebel for ts1


    def Save(self, forcepath = None):
        '''
        Save the actual __dict__ elements in a json, ignoring model and sentiment analysed.
        This means saving the actual state of the DADDBias object into a json so it can be loaded
        later using Load() function
        '''
        if(forcepath is not None):
            filename = "{}/{}".format(self.outputfolder,forcepath)
        else:
            filename = "{}/{}_bias.{}_cluster.{}_USAS.{}.json".format(self.outputfolder,
                self.name,self.performed_bias,self.performed_cluster,self.performed_USAS)
        del self.model
        del self.sid
        js = json.dumps(self.__dict__)
        fp = open(filename, 'w+')
        fp.write(js)
        fp.close()
        return filename
    
    def Load(self, jsonpath):
        '''
        Load the actual __dict__ elements from a json into a DADDBias object.
        '''
        with open(jsonpath) as f:
            x = json.load(f)
            self.__dict__ = x
            self.sid = SentimentIntensityAnalyzer()
            self.model = None
        
    
    def CalculateBiasedWords(self, targetset1, targetset2, minstdev, 
                             acceptedPOS = ['JJ', 'JJS', 'JJR','NN', 'NNS', 'NNP', 'NNPS','VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], 
                             words = None, force=False):
        '''
        This function calculates the list of salient words towards targetset1 and taregset2 with salience > than the 
        specified times (minstdev) the standard deviation.
        
        targetset1 <list of strings> : target set 1 (attribute concept 1)
        targetset2 <list of strings> : target set 2 (attribute concept 2)
        minstdev int : Minium stdev threhsold. used to select most salient words
        acceptedPOS <list<str>> : accepted list of POS to consider for the analysis, as defined in NLTK POS tagging lib. 
                                  If None, no POS filtering is applied and all words in the vocab are considered
        words list<str> : list of words we want to consider. If None all words in the vocab are considered
        force <bool> : forces the execution of this function. 

        >>Returns:
        self.b1_dict : dictionary of words biased towards ts1
        self.b2_dict : dictionary of words biased towards ts2
        
        In which, each word is described as an object like this:
          {'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'rank':rank, 'rankW':rankW, 'sal':val, 'wv':wv.tolist(), 'sent':sent } 
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

        #only keep these words that exist in the model's vocabulary
        self.tset1 = self._keepOnlyModelWords(targetset1)   
        self.tset2 = self._keepOnlyModelWords(targetset2)
        self.stdevs = minstdev
        
        #we remove words in the target sets, and also their plurals from the set of interesting words to process to clean the model's vocab.
        engine = inflect.engine()
        toremove = targetset1 + targetset2 + [engine.plural(w) for w in targetset1] + [engine.plural(w) for w in targetset2]
        if(words is None):
            words = [w for w in self.model.wv.vocab.keys() if w not in toremove]
        
        #calculate centroids for each target set
        self.tset1_centroid = self._calculateCentroid(self.tset1)
        self.tset2_centroid = self._calculateCentroid(self.tset2)
        [minR, maxR] = self._getModelMinMaxRank()
        
        #first estimate word bias towards ts1 and ts2
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
            #filter out non-interesting pos tags
            p = nltk.pos_tag([w])[0][1]
            if acceptedPOS is not None and p not in acceptedPOS:
                continue
            wv = self.model.wv[w]
            sent = self._getSentiment(w)
            freq = self._getWordFreq(w)[1]
            rank = self._getWordFreq(w)[2]
            rankW = 1-self._normalise(rank, minR, maxR) 
            #normalise bias and salience
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
        
        #calculate the salience threshold for both word sets, based on the provided maximum standard deviation
        #parameter stdevs. Finally, select the list of biased words
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

        #transform centroid to list so they become serializable
        self.tset1_centroid = self.tset1_centroid.tolist() 
        self.tset2_centroid = self.tset2_centroid.tolist()
        self.performed_bias = True
        #return dictionaries of salient words (objects)
        return [self.b1_dict, self.b2_dict]
    

    def Clustering(self, repeateachclustering = 100, forcekmin=None, forcekmax=None, kjump=1, force = False):
        '''
        This function clusters similar words in concepts, using k-means over embeddings of the most salient words biased towards ts1 or ts2 (determined in previous function CalculateBiasedWords).
        for each value of k between `forcekmin` and `forcekmax`, this function repeats the k-means clustering `repeateachclustering` times and selects the partition, 
        among all partitions created, based on silhouette value.
        Note that before running this function, the set of most biased words need to be sleected by calling CalculateBiasedWords.
        
        repeateachclustering <int> : Number of times we want to repeat the partitoins for each k to find the best silhouette posisble
        forcekmin <int> : minimum k for kmeans to start exploring 
        forcekmax <int> : maximum k for kmeans to end exploring
        kjump <int> : jump betwen iterations of k
        force <bool> :  forces the execution of this function.

        >>Returns 
        self.clusters1_dict : dictionary of clusters biased towards ts1 with highest silhouette value
        self.clusters2_dict : dictionary of clusters biased towards ts2 with highest silhouette value

        In which each cluster is described by an object which contains
          {'biasW_avg': data[0], 'freq':data[1], 'sent_avg': data[2], 'sal_avg':data[3], 'rankW_avg':data[4], 'centroid':data[5]}
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
        
        #default values to explore    
        self.kstart = 0
        if(forcekmin is not None):
            self.kstart = forcekmin
        self.kend = 200
        if(forcekmax is not None):
            self.kend = forcekmax
        self.krepetitions = repeateachclustering
        
        #prepare list of words and word vectors for both targetsets
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

            #Map the words to each cluster considering the paritition with max silhouette score
            clusters = []
            for i in range(0,len(maxscore[2])):
                cl = []
                indexes = np.where(maxscore[1] == i)[0]
                for idx in indexes:
                    cl.append(words[idx])
                clusters.append(cl)
            #and save the clusters
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
        Auxiliary function that helps populating the list of clusters and estimating their values and cluster averages
        clusters <list<str>> : list of words in each cluster
        wdict <dict> : dictoinary of words which contains information about each word and was created during CalculateBiasedWords

        >>returns
        a dictionary with key = cluster name and value is the cluster object shown below:
            {'biasW_avg': data[0], 'freq':data[1], 'sent_avg': data[2], 'sal_avg':data[3], 'rankW_avg':data[4], 'centroid':data[5]}
        '''
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

    def USASLabels(self, force=False):
        '''
        Returns the most frequent semantic label for all clusters, in the form of a dictionary where the key is the cluster
        name and the value is the most frequent label (or labels)
        '''
        if(not self.performed_cluster or not self.performed_bias):
            raise Exception("you first need to perform the clustering to get the USAS labels")
        if(self.performed_USAS and not force):
            raise Exception("USAS has already been performed in this object. Set force to True to redo the USAS labelling")
        #get usas dicts using USAS_utils library, which connects to the web resource for USAS simulating an http petition to
        #obtain the semantic labels for each cluster.
        self.usasDict1 = USAS_utils.getUsasDict(self.clusters1)
        self.usasDict2 = USAS_utils.getUsasDict(self.clusters2) 
        #update cluster info, including the most frequent semantic label for each cluster
        for k,v in self.usasDict1.items():
            self.clusters1_dict[k]['USAS_label'] = v
        for k,v in self.usasDict2.items():
            self.clusters2_dict[k]['USAS_label'] = v
        self.performed_USAS = True
        #create the most frequent semantic labels rankings for each partition
        self.usasLabelsRanking1 = self._rankUSASLabels(self.clusters1_dict)
        self.usasLabelsRanking2 = self._rankUSASLabels(self.clusters2_dict)
        return [self.usasDict1, self.usasDict2]
    
    def _rankUSASLabels(self, cluster_dict):
        '''
        This method loads a set of clusters and counts how many times the USAS labels appear in the set of clusters.
        Replace for Counter()
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
        
    
    def _orderWordsByFrequency(self, wordL, wDict):
        '''
        Sorts a list of words based on frequency.
        wordl <list<str>> : list of words to order
        wDict biasdcit : the bias dict containing information about the words to order

        >>retusn
        list of words ordered by frequency
        '''
        wl = [ (w, wDict[w]['freq']) for w in wordL ]
        wl.sort(key=lambda x: x[1], reverse=True)
        return [w[0] for w in wl]
        
    
    def GetClusterRanking(self, rankingtype):
        '''
        This function orders the clusters based on a ranking method 'salience', 'frequency', 'sentiment', 'bias', 'rank'.
        rankingtype <str> : 'salience'||'frequency'||'sentiment'|| 'bias'||'rank'

        >>returns
        c1list : list of clusters biased towards ts1 sorted on `rankingtype` sorting strategy
        c2list : list of clusters biased towards ts2 sorted on `rankingtype` sorting strategy
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

        >>returns:
        clusters matching the criteria
        (toremove)
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
        This functions searches the usasDict1 and reassigns the most frequent semantic label to each cluster.
        This was used as an update mechanism.
        (toremove)
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
        Returns the cluster that contains the word `word`or None if any
        (toremove)
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
        '''
        Prints a summary of the DADDBias model.
        (toremove)
        '''
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
    
 
    def _normalise(self, val, minF, maxF):
        '''
        Normalises a value in the positive space
        '''   
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
        This function is used to estimate the threshold to determine which words are salient enough to be considered and which not.

        dword is an object like {'word':w, 'bias':bias, 'biasW':biasW, 'freq':freq, 'freqW':freqW, 'sal':val, 'wv':wv, 'sent':sent }
        stdevs : stdevs fow which we want to compute the threshold
        
        >>returns
        outlier_thr : the threhsold that corresponds with the set quantity of stdevs considering the salience values of the words in  `dwords`
        '''
        allsal = []
        for obj in dwords:
            allsal.append(obj['sal'])
        stdev = statistics.stdev(allsal)
        outlier_thr = (stdev*stdevs)+sum(allsal)/len(allsal)
        return outlier_thr
        