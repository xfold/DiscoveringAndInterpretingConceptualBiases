import requests
import json


def get_USAS_categories_online(text, url='http://ucrel-api.lancaster.ac.uk/cgi-bin/usas.pl',usas_json="USAS.json"):

	d = dict()
	d['email'] = 'a.nobody@here.ac.uk'
	d['tagset'] = 'c7'
	d['style'] = 'vert'
	d['text'] = 'honey beer'

	json_file = open(usas_json, 'r')
	usas_d = json.load(json_file)

	def getUSASCategory(usaskw):
	    '''
	    first preprocess, remove, etc the kw form the website, 
	    since we are only interste din the categories
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


	    return usas_d[aux]


	def parseL(line):
	    #expects a line like this 0000003 010  NN1     honey                    F1 
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
		try:
			#post
			response = requests.post(url, files=d)
			#gather data	
			html = response.text
			#get important part
			cleantext = html[html.index('<pre>')+5: html.index('</pre>')]
			tor = list([])
			#iterate and remove first and last line
			for line in cleantext.split('\n')[2:-2]:
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
	return getUSASOnline(text, url, d)