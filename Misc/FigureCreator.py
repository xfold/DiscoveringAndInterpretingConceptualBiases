from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import itertools
import seaborn as sns
import inflect
from matplotlib import cm
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

    
'''
CLUSTER FIGURE LINEAR PLOT
'''
def CreateLinearPlot(clusters1, clusters2, titles, t1name, t2name, yaxislabel=None, savefilename = None, filterplurals = None, dontinclude = None, nrows=2, ncols=2, figuresize=None):
    '''
    clusters1 <list <list<clusters from DADDBias.py> > : ordered cluster1 lists to display
    clusters2 <list <list<clusters from DADDBias.py> > : ordered cluster2 lists to display
    titles <list<str>> : List of experiments to include in the figure, len(titles) == len(clusters1) == len(clusters2) 
    '''
    #clusters from DADDBias are like this
    #(words, {'biasW_avg': data[0], 'freq':data[1], 'sent_avg': data[2], 'sal_avg':data[3], 'rankW_avg':data[4], 'centroid':data[5], 
    #'USAS_label':USASLABEL})
    
    if(len(titles) != len(clusters1)):
        raise Exception("If you want to perform more than one ranking type, you should include more than one ranking order")
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    if(figuresize is None):
        figuresize = (1+len(clusters1[0])*2*1.3,3*2)
    fig.set_size_inches(figuresize)
    maxfreq = _getMaxFreq(clusters1)
    maxfreq2 = _getMaxFreq(clusters2)
    maxfreq = max(maxfreq, maxfreq2)
    #resize the freqs to the max and min
    maxclusterSize = 7000
    minclusterSize = 150
    index = 0
    for row in ax:
        #this is added for figures with only one column
        if(type(row) is not np.ndarray):
            row = np.array([row])
        for col in row:
            c1 = clusters1[index]
            c2 = clusters2[index]
            #prepare data
            sal1 = np.array([c[1]['sal_avg'] for c in c1])
            sal2 = np.array([c[1]['sal_avg'] for c in c2])
            sents1 = np.array([c[1]['sent_avg'] for c in c1])
            sents2 = np.array([c[1]['sent_avg'] for c in c2])
            if(filterplurals is None or filterplurals == False):
                words1 = np.array([ ['\n'.join(c[0])] for c in c1])
                words2 = np.array([ ['\n'.join(c[0])] for c in c2])
            elif(dontinclude is not None and len(dontinclude)>0):
                words1 = np.array([ ['\n'.join(c)] for c in _selectWords(c1, dontinclude)]) 
                words2 = np.array([ ['\n'.join(c)] for c in _selectWords(c2, dontinclude)]) 
            else:
                words1 = np.array([ ['\n'.join(c)] for c in _selectWords(c1)]) 
                words2 = np.array([ ['\n'.join(c)] for c in _selectWords(c2)]) 
            freq1 = np.array([c[1]['freq'] for c in c1])
            freq2 = np.array([c[1]['freq'] for c in c2])
            #resize freqs so theyre menaingful, normalise freq
            freq1 = freq1 / maxfreq
            freq2 = freq2 / maxfreq    
            freq1 = freq1 * 4**6.5
            freq2 = freq2 * 4**6.5
            freq1 = [f if f<maxclusterSize else maxclusterSize for f in freq1]
            freq2 = [f if f<maxclusterSize else maxclusterSize for f in freq2]
            freq1 = [f if f>minclusterSize else minclusterSize for f in freq1]
            freq2 = [f if f>minclusterSize else minclusterSize for f in freq2]
            print(freq1)
            title = titles[index]
            
            col.scatter(range(0,len(c1)), sal1,   s=freq1, c = sents1, cmap = 'RdYlGn', vmin=-1, vmax=1)
            col.scatter(range(0,len(c2)), sal2*-1,s=freq2, c = sents2, cmap = 'RdYlGn', vmin=-1, vmax=1)
            col.spines['bottom'].set_position('center')
            col.xaxis.set_ticks_position('bottom')
            col.yaxis.set_ticks_position('left')
            col.set_yticklabels( [1.5, 1, 0.5, 0, 0.5, 1,1.5])
            col.spines['right'].set_color('none')
            col.spines['top'].set_color('none')
            col.set_title(title.title(), pad=20, fontsize=14, fontweight='bold')
            
            if(yaxislabel is None):
                yaxislabel = "Bias {}  Bias {}".format(t2name, t1name)
            col.set_ylabel(yaxislabel, fontsize=10)
            col.get_xaxis().set_ticks([])
            col.set_ylim(-1.45,1.45)
            col.tick_params(axis='both', which='major', labelsize=10)
            
            
            for i,(x,y) in enumerate(zip(range(0,len(c1)), sal1)):
                #label = "{:.2f}".format(y)
                col.annotate(', '.join(words1[i]), # this is the text
                             (x,y), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,20), # distance from text to points (x,y)
                             ha='left', # horizontal alignment can be left, right or center
                             fontsize=12) 

            for i,(x,y) in enumerate(zip(range(0,len(c2)), sal2*-1)):
                #label = "{:.2f}".format(y)
                col.annotate(', '.join(words2[i]), # this is the text
                             (x,y), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,-20), # distance from text to points (x,y)
                             ha='left',  # horizontal alignment can be left, right or center
                             fontsize=12)


            
            index+=1
    
      
    if(savefilename is not None):
        print('figure was saved as ', savefilename)
        plt.savefig(savefilename, dpi = 200)
    
    plt.show()
    
def _getMaxFreq(clusters):
    allc = []
    for cl in clusters:
        allc.append( [c[1]['freq'] for c in cl] )
    allc = list(itertools.chain.from_iterable(allc))
    return max(allc)

engine = inflect.engine()
ps = PorterStemmer() 
def _selectWords(words, num=1, dontinclude = None):
    allwused = set(['slutty', "he'sa", "guy'sa"])
    addspacing = ["nazareth", "guilt", "beaten", "cute", "patriarch", "hodgepodge", "threats","baptist","cuck", "trump", "leader"]
    tor = []
    #first remove repetitions and plurals
    for ws in words:
        allwselected = []
        for w in ws[0]:
            if(dontinclude is None):
                plural = engine.plural(w)
                stem = ps.stem(w)
                if(w not in allwused and plural not in allwused and stem not in allwused):
                    if(w in addspacing):
                        w = w+' \n'
                    allwselected.append(w)
                    print('appending ', w, ' plural :', plural, ' stem:', stem)

                allwused.add(w)
                allwused.add(plural)
                allwused.add(stem)
                if(len(allwselected) == num):
                    break
            else:
                if w in dontinclude:
                    pass
                else:
                    if(w=="nazareth" or w == "guilt"):
                        #print('FOUND')
                        w = w+' \n'
                    allwselected.append(w)
        #print(allwselected)
        if(len(allwselected) == 0):
            print('ITS ZERO')
            allwselected.append(ws[0][0])
        tor.append(allwselected[:min(1, len(allwselected))])
    #print(tor)
    return tor
    
'''
PIE
'''
def GeneratePie(clusterplususas1, clusterplususas2, percentthreshold = 0, labelcolors = None, ignoreUnmatched = False, savefilename = None):
    '''
    clusterplususas1 : counting of usas labels as provided by the DADDBIas object 
    clusterplususas2 : counting of usas labels as provided by the DADDBIas object
    percentthreshold <int> : minimum percentage of frequency for a label to be considered
    labelcolors <dictionary<string:string>> : dictionary mapping label name with color
    '''
    #first process the USAS labels and create rankings.
    clusterplususas1 =_rankUSASLabels_modif(clusterplususas1)
    clusterplususas2 =_rankUSASLabels_modif(clusterplususas2)
    
    if(percentthreshold is not None):
        totalC1 = sum( [x[1] for x in clusterplususas1] )
        pc1 = math.ceil(percentthreshold/100 * totalC1)
        clusterplususas1 = [ (x,y) for [x,y] in clusterplususas1 if y >= pc1 and y > 1]
        totalC2 = sum( [x[1] for x in clusterplususas2] )
        pc2 = math.ceil(percentthreshold/100 * totalC2)
        clusterplususas2 = [ (x,y) for [x,y] in clusterplususas2 if y >= pc2 and y > 1]
        print('threshold t1:', pc1, ' th2:', pc2)

    if(ignoreUnmatched is not None and ignoreUnmatched == True):
        ind = [x[0] for x in clusterplususas1].index('Unmatched')
        if(ind>=0):
            print('Removing unmatched t1 ', clusterplususas1[ind])
            del clusterplususas1[ind]
        ind = [x[0] for x in clusterplususas2].index('Unmatched')
        if(ind>=0):
            print('Removing unmatched t2 ', clusterplususas2[ind])
            del clusterplususas2[ind]
        
    f1count = [ x[1] for x in clusterplususas1]
    f1labels = [_rep(x[0]) for x in clusterplususas1]
    f2count = [x[1] for x in clusterplususas2]
    f2labels = [_rep(x[0]) for x in clusterplususas2]
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.tight_layout()
    fig.set_size_inches(11, 3)
    index = 0
    for col in ax:
        if(index == 0):
            count = f1count
            labels = f1labels
        elif(index == 1):
            count = f2count
            labels = f2labels

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
        col.pie(count, labels=labels,autopct='%1.1f%%',shadow=False, startangle=90, colors = colors)
        col.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        index +=1
        #col.title(title)
    if(savefilename is not None):
        print('figure was saved as ', savefilename)
        plt.savefig(savefilename, dpi = 200)
    plt.show()


    
def _rankUSASLabels_modif(cluster_dict):
        '''
        This method does exactly the same as DADDBias _rankUSASLabels method but ignores these unmatched labels that belong
        to clusters with other USAS labels with the same frequency. 
        If a cluster is tagged with USAS = ['unmatched', 'power'], then we consider label (Power) as the cluster label.
        '''
        #Unmatched is never considered as a label if there are other labels as frequent as unmatched in the cluster
        labeldict = {}
        labels = [v['USAS_label'].split(",") for v in cluster_dict.values()]
        #ignore those Unmatched labels that are as frequent as otehr labels, idnicating that the cluster is not unmatched since
        #its labelled
        for ls in labels:
            if(len(ls)>1):
                #contains more than one label
                if('Unmatched' in ls):
                    del ls[ls.index('Unmatched')]
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

def _rep(name):
    #shorten names to display
    toreplace = ['Similar/different', 'and the supernatural', 'and physical properties', 'and writing', 'and personal belongings', 'and Friendliness', 'and related activities', '-', ':-', '(pretty etc.)', ':']
    for r in toreplace:
        if(r in name):
            name = name.replace(r, "")
    return name.lower().capitalize()
    


    
'''
HEATMAP
'''
def GenerateHeatmap(usasLabelsRanking1L, usasLabelsRanking2L, names1L, names2L, size = None, percentthreshold = 0,savefilename = None):
    '''
    usasLabelsRanking1L : List of usasLabelsRankings from DADDBias for targetset1
    usasLabelsRanking2L : List of usasLabelsRankings from DADDBias for targetset2
    names1L list<str> : List of labels for rankings1
    names2L list<str> : List of labels for rankings2
    percentthreshold <int> : ignore labels with lower frequency than the percent threshold wrt the total aggregated label freq
    '''
    usasl1 = []
    usasl2 = []
    #first filter out all these labels non interesting (under freq% threhold), for all sets of labels
    for index in range(0, len(usasLabelsRanking1L)):
        totalC1 = sum( [x[1] for x in usasLabelsRanking1L[index] ] )
        pc1 = math.ceil(percentthreshold/100 * totalC1)
        usasl1.append( [x  for [x,y] in usasLabelsRanking1L[index] if y > pc1 and y > 1])
        totalC2 = sum( [x[1] for x in usasLabelsRanking2L[index] ] )
        pc2 = math.ceil(percentthreshold/100 * totalC2)
        usasl2.append( [x  for [x,y] in usasLabelsRanking2L[index] if y > pc2 and y > 1])
        print('set of labels above thr ({},{}) : ({}, {})'.format(pc1, pc2, len(usasl1[index]), len(usasl2[index])))
    
    usas = usasl1+usasl2
    labels =names1L+names2L
    iM = _getInteresectinMatrix(usas)
    
    #print the matrix
    if(size is None):
        size = (8,8)
    fig = plt.figure(figsize=(size))
    ax = fig.add_subplot(111)
    cax = ax.matshow(iM, interpolation='nearest')
    #ax.set_xticklabels([])
    ax.set_yticklabels([''] + labels)
    ax.set_xticklabels([''] + labels)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = False
        tic.tick2On = True
        tic.label1On = False
        tic.label2On = True
    #Write the avg intersec between usas labels categories
    for i in range(len(iM)):        
        for j in range(len(iM)):
            c = iM[j,i]
            ax.text(i, j, "{0:.2f}".format(c), va='center', ha='center')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if(savefilename is not None):
        print('figure was saved as ', savefilename)
        plt.savefig(savefilename, dpi = 200)
        
    plt.show()
    
def _getInteresectinMatrix(listLabels):
    '''
    listLabels list<list<str>> : list of lists of usas labels. This function
    itareates through all combinations of labels and evaluates their similarity. Returns
    a interesection symmetric matrix of n*n where n = len(listLabels)
    '''
    iM = np.zeros((len(listLabels), len(listLabels)))
    for i in range(0, len(listLabels)):
        s1 = listLabels[i]
        for j in range(0, len(listLabels)):
            s2 = listLabels[j]
            ii = _intersec(s1,s2)
            iM[i][j] = ii
    return iM        
    
def _intersec(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    return float(intersection) / min(len(list1),len(list2))
    
