import json
import requests
from operator import itemgetter


def getUsasDict(clusters, ignore_USAS_label = []):
    '''
    clusters <list <str>> : list of words in every cluster
    ignore_USAS_label <list<str>> : list of usas labels to ignore
    '''
    topusaslabels = 1 #we keep top most frequent label (or labels), since we are filtering for [Unmatched','Grammatical bin'] categories
    usas_output1 = LabelClusters_USAS(clusters)
    usas_output1 = CleanUSAS(usas_output1, topusaslabels, ignore_USAS_label)
    
    cl1_indexlabelUSAS = {}
    for i, k in enumerate(usas_output1):
        clustername = ' '.join(k[1])
        labels = k[0]
        
        labs = [ lab[0][0].split(',')[0] for lab in labels]
        cl1_indexlabelUSAS[clustername] = ', '.join(labs)

    return cl1_indexlabelUSAS


def LabelClusters_USAS(clusters, pathToUsasJson = 'Source/Misc/USASSemtags/USAS.json'):
    auxL = []
    for c in clusters:
        strc = ' '.join(c)
        response = _get_USAS_categories_online2(strc, usas_json=pathToUsasJson)
        usas_cat = _accCategories_USAS(response)
        auxL.append( ([ usas_cat ], c) )
    return auxL


def _get_USAS_categories_online2(text, url='http://ucrel-api.lancaster.ac.uk/cgi-bin/usas.pl',usas_json="USASSemtags/USAS.json"):
    '''
    Add this type = web 10.8.2019
    '''
    d = dict()
    d['email'] = 'a.nobody@here.ac.uk'
    d['tagset'] = 'c7'
    d['style'] = 'vert'
    d['type'] = 'web'
    d['text'] = 'honey beer'
    json_file = open(usas_json, 'r')
    usas_d = json.load(json_file)
    def getUSASCategory(usaskw):
        '''
        first preprocess, remove, etc the kw
        '''
        aux = usaskw.strip()
        if '[' in aux:
            aux = aux[0:aux.index('[')]
        if 'm' in aux:
            aux = aux[0:aux.index('m')]
        if 'n' in aux:
            aux = aux[0:aux.index('n')]
        if 'f' in aux:
            aux = aux[0:aux.index('f')]
        if 'c' in aux:
            aux = aux[0:aux.index('c')]
        if '&' in aux:
            aux = aux[0:aux.index('&')]
        if '+' in aux:
            aux = aux[0:aux.index('+')]
        if '%' in aux:
            aux = aux[0:aux.index('%')]
        if '@' in aux:
            aux = aux[0:aux.index('@')]
        if '-' in aux:
            aux = aux[0:aux.index('-')]
        if '/' in aux:
            aux = aux[0:aux.index('/')]
        
        #print(aux)
        #print(usas_d[aux])
        
        return usas_d[aux]
    def parseL(line):
        #expects a lien like this 0000003 010  NN1     honey                    F1 
        auxl = list([])
        cw = list([])
        for c in line:
            if c!=' ' or len(auxl)== 4:
                cw.append(c)
            elif c == ' ' and len(cw) >0:
                auxl.append(''.join(cw))
                cw = []
        
        usascat = ''.join(cw).strip().split(' ')
        r = []
        for c in usascat:
            r.append( (c, getUSASCategory(c)) )
            
        #print('categoreis {}'.format(r))
            
        auxl.append(r)
        return auxl
    def getUSASOnline(text, url='http://ucrel-api.lancaster.ac.uk/cgi-bin/usas.pl', dictionary = d):
        '''
        returns a list of [[w, pos, usas], ...] for every w in text.
        in which:
          w : word
          pos : pos tag as for USAS online categoriser
          usas = [[cat1],[cat2],[cat3]] : usas category. list may contain more than one category per word!
        '''
        d['text'] = text
        tor = list([])
        #print('words to tag {}'.format(text))
        try:
            #post
            response = requests.post(url, files=d)
            #gather data    
            html = response.text
            #get important part
            #print(html)
            '''
            Add this! 10.08.2019
            '''
            #print('response : {}'.format(html) )
            if '<pre>' in html:
                cleantext = html[html.index('<pre>')+5: html.index('</pre>')]
            else:
                cleantext = html

            tor = list([])
            #iterate and remove first and last line
            #print('cleantext split {}'.format(cleantext.split('\n')))
            #print('cleantext split {}'.format(cleantext.split('\n')[2:-]))
            for line in cleantext.split('\n')[2:-1]:
                try:
                    wl = line.strip()
                    [ids, x, pos, w, usas] = parseL(wl)
                    tor.append( [w, pos, usas] )
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
        return tor
    #runrun
    #print('HEY TAHTS IT')
    return getUSASOnline(text, url, d)


'''
USAS code for caggregating categoreis
'''
def UpdateCount_USAS(key, v, dic):
    if key in dic:
        dic[key].append(v)
    else:
        dic[key] = [v]
    return dic #no need but meh

def _accCategories_USAS(usas_l):
    '''
    This function takes as an input the get_USAS_categories_online ouptut, which consists of a list of lists like this
    [[w, pos, usas], ...]   // w : word, pos : part of speech (nos used), usas: usas categorisation
        
    Later, this function aggregates all words with similar categories ad orders them. 
    >> returns
    [
        [ category name , list of words in that category , len( list of words ) ],
        [ category2 name , list of words in that category2 , len( list of words 2 ) ],
        ...
    ]
    '''
    di = dict()
    for l in usas_l:
        els = l[2]
        w = l[0]
        for e in els:
            key = e[1]
            di = UpdateCount_USAS(key, w, di )
    
    l = [(k, v, len(v)) for k, v in di.items()]
    l =sorted(l, reverse=True,key=itemgetter(2))
    
    return l


def CleanUSAS(usas_ouput, topk, ignore_USAS_label = []):
    ret = []
    
    for cluster in usas_ouput:
        cset = set([])
        cret = []
        
        for usas_label in cluster[0][0]:
            
            lname = usas_label[0]
            if(lname in ignore_USAS_label):
                continue

            lcount = usas_label[2]
            cset.add(lcount)
            if(len(cset)>topk):
                break
            else:
                cret.append( [(lname, lcount)] )
        
        ret.append([cret, cluster[1]])
        
    return ret
