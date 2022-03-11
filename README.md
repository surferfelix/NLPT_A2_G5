# NLPT_A2_G5
Group Assignment for the Course NLP Technology at VU University Amsterdam

## AUTHORS
------------------
A.M. Dobrzeniecka (Alicja), E.H.W. Galjaard (Ellemijn), F. den Heijer (Felix), S.Shen (Shuyi)

## PROJECT STRUCTURE
-------------------
This project tackles the NLP task of Semantic Role labeling (SRL) with rule based and traditional along with neural machine learning approaches. Semantic roles involve identification of

**What you will find in this project:**
- _A rule based system for predicate and argument identification_
- _Traditional ML systems for predicate and argument identification, along with argument classification_
- _A neural SRL system that can tackle SRL on its own_

## PREDICATE AND ARGUMENT IDENTIFICATION AND ARGUMENT CLASSIFICATION

### IMPLEMENTED FEATURES

- tokens
- indices
- lemmas
- universal Part-of-Speech tags
- language-specific Part-of-Speech tags
- morphological features
- the ID of the head word
- the universal dependency relation to the head
- word, as well as head ID
- dependency relation pairs

### HOW TO USE IT
- STEP 1. Install libraries specified in **requirements.txt** by running    
    `pip install -r requirements.txt`    
- STEP 2. Run main.py with proper argument. We have three arguments that have to be provided:
    - decide if you want to run rule-based or svm-based identification by stating **'yes'** (rule-based) or **'no'** (svm-based)
    - decide if you want to run the mini version of data or the full one by stating **'yes'** (if mini) or **'no'** (if full)
    - decide if you want to use embedding as a feature in svm model by stating **'yes'** (with embedding) or **'no'** (without embedding)
    - provide the path to the embedding model **your_model_path** (you can download the text model from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)

The examplary command would be:
- **python3 main.py 'no' 'no' 'no'**

## A NEURAL SRL SYSTEM 

The code for the SRL system is stored in the same repositorium. 

### HOW TO USE IT
- STEP 1. Install libraries specified in **requirements.txt** by running    
    `pip install -r requirements.txt`    
- STEP 2. Go to **srl_assignment_code/** folder and run **srl_main.py**
