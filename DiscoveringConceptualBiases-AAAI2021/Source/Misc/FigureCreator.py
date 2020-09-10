'''
******************************************************************************************************************
NOTE: Although the code has been cleaned and commented, it is far from a final version. Many structures and procedures could be improved. 
We are still working on it, but hopefully this serves as a beta version to demo the approach presented in the paper.
******************************************************************************************************************
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import itertools
import inflect
from nltk.stem import PorterStemmer 

    
'''
This file contains a set of functions to create the figures shown in the paper based on the reuslts obtained by the approach.
The three main functions are:
 - CreateLinearPlot : Function to generate the detailed rankings of conceptual biases shown in the paper
 - GeneratePie : Function to generate the pies that show the conceptual biases of the community
 - GenerateHeatmap : function to generate the heatmaps shown in the paper
'''
def CreateLinearPlot(clusters1, clusters2, titles, t1name, t2name, yaxislabel=None, savefilename = None, filterplurals = None, nrows=2, ncols=2, figuresize=None):
    '''
    Creates a linear plot showing the detailed rankings of clusters presented in the paper. 

    clusters1 <list <list<clusters from DADDBias.py> > : ordered cluster1 lists to display. 
    clusters2 <list <list<clusters from DADDBias.py> > : ordered cluster2 lists to display. 
    titles <list<str>> : List of experiments to include in the figure, len(titles) == len(clusters1) == len(clusters2) 
    t1name <str> : name for the tagret set1 y-axis, will be combined with `t2name`
    t2name <str> : name for the tagret set2 y-axis, will be combined with `t1name`
    yaxislabel <str> : full y axis name. Leave as None if you want to use `t1name` and `t2name`. (remove)
    savefilename <str> : Path to save the resulting figure
    filterplurals <bool> : set to True to ignore clusters formed by plurals already shown before. This is helpful to save space to generate smaller and more informative figures .
    nrows <int> : Number of rows of the resulting image - useful if we want to leave more space between clusters, by i.e. setting 4 rows and 1 column. Nrows*ncols should be equal to len(clusters1) and to len(clusters2)
    ncols <int> : Number of cols of the resulting image. Nrows*ncols should be equal to len(clusters1) and to len(clusters2) 
    figuresize (<int>, <int>) : force figure size
    '''
    
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
    #resize the freqs to the max and min to beautify the results based on freqs
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
            title = titles[index]
            
            #plot and misc
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
                col.annotate(', '.join(words1[i]),          # this is the text
                             (x,y),                         # this is the point to label
                             textcoords="offset points",    # how to position the text
                             xytext=(0,20),                 # distance from text to points (x,y)
                             ha='left',                     # horizontal alignment can be left, right or center
                             fontsize=12) 

            for i,(x,y) in enumerate(zip(range(0,len(c2)), sal2*-1)):
                col.annotate(', '.join(words2[i]),          # this is the text
                             (x,y),                         # this is the point to label
                             textcoords="offset points",    # how to position the text
                             xytext=(0,-20),                 # distance from text to points (x,y)
                             ha='left',                     # horizontal alignment can be left, right or center
                             fontsize=12) 
            index+=1
    
      
    if(savefilename is not None):
        print('figure was saved as ', savefilename)
        plt.savefig(savefilename, dpi = 200)
    
    plt.show()
    
   
'''
PIE
'''
def GeneratePie(clusterplususas1, clusterplususas2, percentthreshold = 0, labelcolors = None, ignoreUnmatched = True, savefilename = None):
    '''
    Creates a pie chart showing the distribution of semantic labels among the set of clusters biased towards ts1 and ts2. `Unmatched' is not considered
    a semantic category, therefore `ignoreUnmatched` is set to True by default.

    clusterplususas1 : counting of usas labels as provided by the DADDBIas object 
    clusterplususas2 : counting of usas labels as provided by the DADDBIas object
    percentthreshold <float> : minimum percentage of frequency for a label to be considered in the pie. 
    labelcolors <dictionary<string:string>> : dictionary mapping cluster label name with color
    ignoreUnmatched <bool> : True/False indicating if we want to show unmatched clusters in the pie.
    savefilename <str> : Path to save output pie image
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

    if(ignoreUnmatched is not None and ignoreUnmatched == True):
        ind = [x[0] for x in clusterplususas1].index('Unmatched')
        if(ind>=0):
            del clusterplususas1[ind]
        ind = [x[0] for x in clusterplususas2].index('Unmatched')
        if(ind>=0):
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
    if(savefilename is not None):
        print('figure was saved as ', savefilename)
        plt.savefig(savefilename, dpi = 200)
    plt.show()

    
'''
HEATMAP
'''
def GenerateHeatmap(usasLabelsRanking1L, usasLabelsRanking2L, names1L, names2L, size = None, percentthreshold = 0,savefilename = None):
    '''
    Creates a heatmap showing the intersection between all semantic labels of the different models above a relative frequency threshold.

    usasLabelsRanking1L : List of usasLabelsRankings from DADDBias for targetset1
    usasLabelsRanking2L : List of usasLabelsRankings from DADDBias for targetset2
    names1L list<str> : List of labels for rankings1
    names2L list<str> : List of labels for rankings2
    size : size of the plot
    percentthreshold <float> : ignore labels with lower frequency than the percent threshold wrt the total aggregated label freq
    savefilename <str> : Path to save output pie image
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
    usas = usasl1+usasl2
    labels =names1L+names2L
    iM = _getInteresectinMatrix(usas)
    
    #print the matrix
    if(size is None):
        size = (8,8)
    fig = plt.figure(figsize=(size))
    ax = fig.add_subplot(111)
    cax = ax.matshow(iM, interpolation='nearest')
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

'''
Aux functions    
'''
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
    
def _getMaxFreq(clusters):
    allc = []
    for cl in clusters:
        allc.append( [c[1]['freq'] for c in cl] )
    allc = list(itertools.chain.from_iterable(allc))
    return max(allc)

engine = inflect.engine()
ps = PorterStemmer() 
def _selectWords(words, num=1):
    '''
    Filters out clusters with repeated stems to show a larger variety of biases in the limited space of the figure.
    Also cleans parsing and stemming mistakes, and adds spacing to specific words to improve visualisation.
    words <list<str>> : words to process
    num <int> : number of words to process (since we do not use all words as a cluster label, we don't process them all)

    >>returns
    The list of words already processed and with added spacing if needed
    '''
    allwused = set(["slutty", "he'sa", "guy'sa"])
    addspacing = ["nazareth", "guilt", "beaten", "cute", "patriarch", "hodgepodge", "threats","baptist","cuck", "trump", "leader"]
    tor = []
    #first remove repetitions and plurals
    for ws in words:
        allwselected = []
        for w in ws[0]:
            plural = engine.plural(w)
            stem = ps.stem(w)
            if(w not in allwused and plural not in allwused and stem not in allwused):
                if(w in addspacing):
                    w = w+' \n'
                allwselected.append(w)
            allwused.add(w)
            allwused.add(plural)
            allwused.add(stem)
            if(len(allwselected) == num):
                break
        if(len(allwselected) == 0):
            allwselected.append(ws[0][0])
        tor.append(allwselected[:min(1, len(allwselected))])
    return tor

def _rankUSASLabels_modif(cluster_dict):
        '''
        This method does exactly the same as DADDBias _rankUSASLabels method but ignores these unmatched labels that belong
        to clusters with other USAS labels with the same frequency. For instance, if a cluster is tagged with USAS = ['unmatched', 'power'] 
        (meaning that both labels have the same freuency in the cluster), then we consider label (Power) as the cluster label.
        '''
        #Unmatched is never considered as a label if there are other labels as frequent as unmatched labels in the cluster
        labeldict = {}
        labels = [v['USAS_label'].split(",") for v in cluster_dict.values()]
        #ignore those Unmatched labels that are as frequent as otehr labels, idnicating that the cluster is actually labelled
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
    '''
    Shorten names to display in the figures.
    '''
    toreplace = ['Similar/different', 'and the supernatural', 'and physical properties', 'and writing', 'and personal belongings', 'and Friendliness', 'and related activities', '-', ':-', '(pretty etc.)', ':']
    for r in toreplace:
        if(r in name):
            name = name.replace(r, "")
    return name.lower().capitalize()
    
