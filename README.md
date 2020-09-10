# Discovering and Interpreting Conceptual Biases in Online Communities Source Code - AAAI2021 - Blind version
This repository contains the source code of the original paper `Discovering and Interpreting Conceptual Biases in Online Communities` submitted to The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21).

<b>Abstract</b>
Language carries implicit human biases, functioning both as areflection and a perpetuation of stereotypes that people carrywith them. Recently, ML-based NLP methods such as wordembeddings have been shown to learn such language biasesstrikingly accurately. This capability of word embeddings hasbeen successfully exploited as a tool to quantify and study hu-man biases. However, previous studies only consider a prede-fined set of conceptual biases to attest (e.g., whether gender ismore or less associated with particular jobs), or just discoverbiased words without helping to understand their meaning atthe conceptual level. As such, these approaches are either un-able to find conceptual biases that have not been defined inadvance, or the biases they find are difficult to interpret andstudy. This makes existing approaches unsuitable to discoverand  interpret  biases  in  online  communities,  as  online  com-munities may have different biases from mainstream culturewhich need to be discovered and properly interpreted. Thispaper proposes a general, data-driven approach to automati-cally discover and help interpret conceptual biases encoded inword embeddings. We apply this approach to study the con-ceptual biases present in the language used in online commu-nities and experimentally show the validity and stability ofour method

# Overview
This repository contains the next files and folders:
<ul>
  <li><b>Datasets/</b>: Folder containing the toy dataset <i>toy_1000_trp.csv</i>. Other datasets should be downloaded (see below).</li>
  <li><b>Models/</b>: Folder containing the toy model for dataset <i>toy_1000_trp.csv</i>, together with DADDobject pretrained models (see below).</li>
  <li><b>Source/</b>: Folder containing the source code used in this submission.  </li>
  <li><b>Source/Misc</b>: Folder containing helper functions related with the submission.</li>
  <li><b>Run.example.train.py</b>: Training example which used the toy dataset provided in this repository to train an embeddings model. </li>
  <li><b>Run.example.bias.py</b>: Apply our method to discover the biases of the toy example model trained with `Run.example.train.py`. </li>
  <li><b>Run.example.bias.py</b>: Python notebook full example utilising the toy example which includes: (i) training an embedding model, (ii) discovering biases and clustering, and (iii) exploration of the biases of the model.</li>
  <li><b>requirements.txt</b>: Requirements file.</li>
  <li><b>README.md</b>: This file.</li>    
</ul>

