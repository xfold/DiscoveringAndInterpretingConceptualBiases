# Discovering and Interpreting Conceptual Biases in Online Communities Source Code - AAAI2021 - Blind version
This repository contains the source code of the original paper `Discovering and Interpreting Conceptual Biases in Online Communities` submitted to The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21).

<b>Abstract</b>
Language carries implicit human biases, functioning both as areflection and a perpetuation of stereotypes that people carry with them. Recently, ML-based NLP methods such as word embeddings have been shown to learn such language biases strikingly accurately. This capability of word embeddings has been successfully exploited as a tool to quantify and study human biases. However, previous studies only consider a predefined set of conceptual biases to attest (e.g., whether gender is more or less associated with particular jobs), or just discover biased words without helping to understand their meaning at the conceptual level. As such, these approaches are either unable to find conceptual biases that have not been defined inadvance, or the biases they find are difficult to interpret and study. This makes existing approaches unsuitable to discover and interpret  biases  in  online  communities,  as  online  communities may have different biases from mainstream culture which need to be discovered and properly interpreted. This paper proposes a general, data-driven approach to automatically discover and help interpret conceptual biases encoded in word embeddings. We apply this approach to study the conceptual biases present in the language used in online communities and experimentally show the validity and stability of our method

# Overview
This repository contains the next files and folders:
<ul>
  <li><b>Datasets/</b>: Folder containing the toy dataset <i>toy_1000_trp.csv</i>. Other datasets should be downloaded (see below).</li>
  <li><b>Models/</b>: Folder containing the toy model for dataset <i>toy_1000_trp.csv</i>, together with the pretrained models used in the paper.</li>
  <li><b>Source/</b>: Folder containing the source code used in this submission.  </li>
  <li><b>Source/Misc</b>: Folder containing helper functions related with the submission.</li>
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
This command will train a model for a small toy dataset collected from TheRedPill included in the project, estimate its gender biases towards women and men, cluster them in concepts, assign a semantic label to each cluster and finally save all calculations and discovered conceptual biases in a json file to facilitate the analysis and creation of figrues. <b>Note that this is a toy dataset created with just a very small part of the original dataset, used to only test the approach, and trained using a reduced set of parameters to speed up the testing process. The results obtained with this toy dataset might not correspond or be simliar to the ones presented in the paper. The full datasets used in this work are presented next. </b>

Another possibility is to run the python notebook file (`Run.example.full.ipynb`) included in the project in which we present an step by step explanation of the process and results.
