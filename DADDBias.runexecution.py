from importlib import reload
import DADDBias
reload(DADDBias)

# attribute concepts for r/trp
female=["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
male=["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"] 

# attribute concepts for r/atheism
islam = ["allah", "ramadan", "turban", "emir", "salaam", "sunni", "koran", "imam", "sultan", "prophet", "veil", "ayatollah", "shiite", "mosque", "islam", "sheik", "muslim", "muhammad"]
christian = ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation", "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross", "catholic", "church"]

#
# Configurations.
#
# add other configs here to train various models
# (note this is a toy training example to test the training process, the parameters used here are different from those used in the paper!)
#
allconfigs = [
    {'modelfile': 'EmbeddingModels/toy_w4_f10_e100_d100',  'output': 'DADDBiasModels/', 'name': 'toy_w4_f10_e100_d100',
     'stdev':4, 't1': female, 't2': male, 'repeatk':30, 'mink':10, 'maxk':30  }
]

#
# Find the biases and save the final bias model
#
for config in allconfigs:
    print('>>starting with config', config)
    obj = DADDBias.DADDBias(config['modelfile'], config['output'], config['name'])
    print('>>calculating bias')
    obj.CalculateBiasedWords(config['t1'], config['t2'], config['stdev'] )
    print('>>clustering')
    obj.Clustering(repeateachclustering = config['repeatk'], forcekmin= config['mink'], forcekmax = config['maxk'])
    print('>>usas labels')
    obj.USASLabels()
    path = obj.Save()
    print(">Config saved in ",path)