from Source import TrainModels

#
# Training Configuration
# 
# add here the different training configs to train different models
# (note this is a toy training example to test the training process, the parameters used are different from those used in the paper!)
trainingSetup = [
    {'csvfile': "Datasets/toy_1000_trp.csv", 'outputFile': 'Models/toy_w4_f10_e100_d200', 'w':4, 'minf': 10, 'epochs':50 ,'ndim':200}  
]

#
# Run the training process
#
for setup in trainingSetup:
    print('Print setup ', setup)
    TrainModels.TrainModel(setup['csvfile'], 
       'body\r',
       outputname = setup['outputFile'],
       window = setup['w'],
       minf = setup['minf'],
       epochs = setup['epochs'],
       ndim = setup['ndim'],
       encoding = "ISO-8859-1" #change encoding to default utf-8 for other csvs
       )