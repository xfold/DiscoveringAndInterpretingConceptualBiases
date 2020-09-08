import TrainModels

trainingSetup = [
    #different dimensions
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f10_e100_d100', 'w':4, 'minf': 10, 'epochs':100 ,'ndim':100},
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f10_e100_d200', 'w':4, 'minf': 10, 'epochs':100 ,'ndim':200},
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f10_e100_d300', 'w':4, 'minf': 10, 'epochs':100 ,'ndim':300},
    #different windows
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w3_f10_e100_d200', 'w':3, 'minf': 10, 'epochs':100 ,'ndim':200},
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w5_f10_e100_d200', 'w':5, 'minf': 10, 'epochs':100 ,'ndim':200},
    #different epochs 
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f10_e200_d200', 'w':4, 'minf': 10, 'epochs':200 ,'ndim':200},
    #different thresholds
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f100_e100_d200', 'w':4, 'minf': 100, 'epochs':100 ,'ndim':200},
    {'csvfile': "C:/Corpora/subreddits/subreddits/theredpill/comments.csv", 'outputFile': '../Models/trp_w4_f1000_e100_d200', 'w':4, 'minf': 1000, 'epochs':100 ,'ndim':200}
]


for setup in trainingSetup:
    print('Print setup ', setup)
    TrainModels.TrainModel(setup['csvfile'], 
                           'body',
                           outputname = setup['outputFile'],
                           window = setup['w'],
                           minf = setup['minf'],
                           epochs = setup['epochs'],
                           ndim = setup['ndim'],
                           )