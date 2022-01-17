import gensim 
import pandas as pd
import time


def TrainModel(csv_document, csv_comment_column='body', outputname='outputModel', window = 4, minf=10, epochs=100, ndim=200, lemmatiseFirst = False, encoding = "utf-8"):
    '''
    Load the documents from document_l, a list of sentences, and train a WE model with specified
    minf, epochs and ndims. where:
    csv_document : csv document containing all information, where each comment is on a different row
    csv_comment_column : name of the column taht contains the text we want to process
    outputname : output path of the resulting model
    
    returns
    path of the trained models
    '''
    
    def loadCSVAndPreprocess(path, column = 'body', nrowss=None, encoding="utf-8"):
        trpCom = pd.read_csv(path, lineterminator='\n', nrows=nrowss, encoding = encoding)
        '''
         read the tokenized reviews into a list
         each review item becomes a series of words
         so this becomes a list of lists
         documents = [gensim.utils.simple_preprocess (line) for line in trpCom['body']]
         
         if nrowss == None, we process the whole excel
        '''
        print('total rows {}'.format(len(trpCom[column])))

        documents = []
        for i, row in enumerate(trpCom[column]):
            if i%500000 == 0:
                print('\t...processing line {}'.format(i))

            try:
                pp = gensim.utils.simple_preprocess (row)
                if(lemmatiseFirst == True):
                    pp = [wordnet_lemmatizer.lemmatize(w, pos="n") for w in pp]
                documents.append(pp)
            except:
                pass
                #print('\terror with row {}'.format(row))


        print("Done reading and preprocessing data file {} ".format(path))
        return documents

    def trainWEModel(documents, outputfile, ndim, window, minfreq, epochss):
        '''
        size
        The size of the dense vector to represent each token or word. If you have very limited data, then size should be a much smaller
        value. If you have lots of data, its good to experiment with various sizes. A value of 100-150 has worked well for me.

        window
        The maximum distance between the target word and its neighboring word. If your neighbor's position is greater than the maximum 
        window width to the left and the right, then, some neighbors are not considered as being related to the target word. In theory, a 
        smaller window should give you terms that are more related. If you have lots of data, then the window size should not matter too 
        much, as long as its a decent sized window.

        min_count
        Minimium frequency count of words. The model would ignore words that do not statisfy the min_count. Extremely infrequent words are 
        usually unimportant, so its best to get rid of those. Unless your dataset is really tiny, this does not really affect the model.

        workers
        How many threads to use behind the scenes?
        '''
        starttime = time.time()
        print('->->Starting training model {} with dimensions:{}, minf:{}, epochs:{}'.format(outputfile,ndim, minfreq, epochss))
        model = gensim.models.Word2Vec (documents, size=ndim, window=window, min_count=minfreq, workers=5, sg=1)
        model.train(documents,total_examples=len(documents),epochs=epochss)
        model.save(outputfile)
        print('->-> Model saved in {}'.format(outputfile))

        
    
    print('->Starting with {} [{}], output {}, window {}, minf {}, epochs {}, ndim {}'.format(csv_document, 
                                                                                       csv_comment_column,
                                                                                       outputname, window, minf, epochs, ndim))
    docs = loadCSVAndPreprocess(csv_document, csv_comment_column, None, encoding)
    starttime = time.time()
    ofile = outputname
    print('-> Output will be saved in {}'.format(ofile))
    trainWEModel(docs, ofile, ndim, window, minf, epochs)
    print('-> Model creation ended in {} seconds'.format(time.time()-starttime))
    
