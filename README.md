# Discovering and Interpreting Conceptual Biases in Online Communities Source Code
This repository contains the source code of the original paper `Discovering and Interpreting Conceptual Biases in Online Communities`.
This work is part of the project [Discovering and Attesting Digital Discrimination (DADD)](https://dadd-project.github.io/). 
Related to this work, we created the [Language Bias Visualiser](https://xfold.github.io/WE-GenderBiasVisualisationWeb/), an interactive web-based platform that helps exploring gender biases found in various Reddit datasets.

<b>Abstract</b>
Language carries implicit human biases, functioning both as areflection and a perpetuation of stereotypes that people carry with them. Recently, ML-based NLP methods such as word embeddings have been shown to learn such language biases strikingly accurately. This capability of word embeddings has been successfully exploited as a tool to quantify and study human biases. However, previous studies only consider a predefined set of conceptual biases to attest (e.g., whether gender is more or less associated with particular jobs), or just discover biased words without helping to understand their meaning at the conceptual level. As such, these approaches are either unable to find conceptual biases that have not been defined inadvance, or the biases they find are difficult to interpret and study. This makes existing approaches unsuitable to discover and interpret  biases  in  online  communities,  as  online  communities may have different biases from mainstream culture which need to be discovered and properly interpreted. This paper proposes a general, data-driven approach to automatically discover and help interpret conceptual biases encoded in word embeddings. We apply this approach to study the conceptual biases present in the language used in online communities and experimentally show the validity and stability of our method

# Overview
This repository contains the next files and folders:
<ul>
  <li><b>Datasets/</b>: Folder containing the toy dataset <i>toy_1000_trp.csv</i>. Other datasets should be downloaded (see below).</li>
  <li><b>Models/</b>: Folder containing the toy model for dataset <i>toy_1000_trp.csv</i>, together with the pretrained models used in the paper.</li>
  <li><b>Source/</b>: Folder containing the source code used in this submission.  </li>
  <li><b>Source/Misc</b>: Folder containing helper functions related with the submission.</li>
  <li><b>PaperExperiments/</b>: Folder containing a jupyter notebook with a step by step explanation showing how to reproduce all figures presented in the paper.
  <li><b>Run.example.train.py</b>: Training example which used the toy dataset provided in this repository to train an embeddings model. </li>
  <li><b>Run.example.bias.py</b>: Apply our method to discover the biases of the toy example model trained with `Run.example.train.py`. </li>
  <li><b>Run.example.full.ipynb</b>: Python notebook walkthrough exemplifying the whole bias discovery process. This includes: (i) training an embedding model, (ii) discovering biases and clustering, (iii) semantic label tagging, and (iv) exploration of the biases of the model using the toy dataset provided in this project.</li>
  <li><b>requirements.txt</b>: Requirements file.</li>
  <li><b>README.md</b>: This file.</li>    
</ul>

# Setup
First, you need to install all dependencies and libraries:
```python
pip3 install -r requirements.txt
```
Now we are ready to run a toy experiment with the provided Dataset and see if everything is working (Python 3):
```python
python3 Run.example.train.py
python3 Run.example.bias.py
```
This command will train a model for a small toy dataset collected from TheRedPill included in the project, estimate its gender biases towards women and men attribute sets, cluster them in concepts, assign a semantic label to each cluster and finally save all calculations and discovered conceptual biases in a json file to facilitate the analysis and the creation of figures. <i>Note that this is a toy dataset created with just a very small part of the original dataset, used to only test the approach. The training, bias parameters, and results obtained with this toy dataset might not be simliar to the ones presented in the paper. To reproduce the paper results, please use the jupyter notebook found in `PaperExperiments/` folder. Also, the original Reddit datasets used in this work are introduced next. </i>

Another possibility is to run the python notebook file `Run.example.full.ipynb` included in the project in which we present an step by step explanation of the process and results.

# Experiments and Reproducibility

## Datasets and Model Training
The original Reddit datasets used in the paper can be downloaded [here](https://osf.io/qmf62/?view_only=6be755746530433da0a5d985ffa69579). Note they are large!
Once the datasets are downloaded, you can train the same models used in the paper by following the examples presented in the python notebook `Run.example.full.ipynb` - make sure you use the same parameters reported in the paper.   

## Bias models
Since training of the embedding models and the discovering of biases is a slow process, we are also providing the results of executing our methodology on the original datasets in a json format. The Bias models can be loaded using the `DADDBias.py` class, and contain all information related to the bias discovery, clustering and semantic tagging of the biases of a community. The Bias models can also be used to generate the Figures presented in the paper, as shown in the jupyter notebook included in the folder `PaperExperiments/`.

Bias models can be found in folder `Models/` and can be loaded using `DADDBias.Load()` function. For instance, to load the Bias model for r/TheRedPill we used in the paper, we need to:
```python
import DADDBias

savedfile = 'Models//toy_w4_f10_e100_d200_bias_bias.True_cluster.True_USAS.True.json'
test = DADDBias.DADDBias()
test.Load(savedfile)
```

Once a model is loaded, we can access all information regarding that execution (see `Source/DADDBias.py` object). Some of the most relevant are:
```python
#biased and most salient words
b1_dict             #dictionary of biased words towards target set 1
b2_dict             #dictionary of biased words towards target set 2
#conceptual biases
clusters1_dict      #cluster dicitonary for ts1
clusters2_dict      #cluster dicitonary for ts2
#semantic categorisation of conceptual biases 
usasLabelsRanking1  #agrgegation of USAS labels at a partition lebel for ts1
usasLabelsRanking2  #agrgegation of USAS labels at a partition lebel for ts1
```

Therefore, once a bias model is loaded, the variable `b1_dict` will contain the set of most salient words selected biased towards attribute concept 1. For example, with next code we are listing all selected salient words biased towards attribute concept 1 (`women` in this model):
```python
print('Total select words biased towards women:', len(test.b1_dict))
for k,v in list(test.b1_dict.items()):
    print(k, '\t (salience ', test.b1_dict[k]['sal'], ')')
```
Or we can also access the set of conceptual biases by querying `clusters2_dict`. For example, with next code we are listing all clusters biased towards attribute concept 2 (`men` in this model):
```python
print('Total conceptual clusters biased towards men: ',len(test.clusters2))
for k in test.clusters2:
    print(k)
```

## Reproducibility
By providing the original Datasets, the source code to train your models and the resulting Bias models of the executions used in the paper, we want to make sure the results are easily reproducible. we are also including a jupyter notebook in folder `PaperExperiments/` which shows how we generated all paper figures, and that is ready to run by anyone with a jupyter notebook installed on their computer. 

# Contact
You can find us on our website on [Discovering and Attesting Digital Discrimination](http://dadd-project.org/), or at [@DADD_project](https://twitter.com/DADD_project).
<br>
<i>Updated Oct 2020</i>

