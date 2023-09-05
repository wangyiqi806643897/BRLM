# BRLM

#### Introduction
BRLM, short for BioBERT vectorized ResNet classified Language processing Model, designed for pathogenicity classification of SNVs in annotation texts.

#### Software Architecture
There are two files with two steps functions.
1.  BRLM trained on 12 TCGA cancers
2.  BRLM successfully derived classification outcomes for CCM familial WGS variants.


#### 

Installation

1.  conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
2.  install.packages("PMAPscore")
3.  BiocManager::install(c("clusterProfiler","DOSE","org.Hs.eg.db","umap"))

#### 
Instructions

1.  Open the corresponding folder based on required dataset (TCGA or CCM)
2.  All-in-one script: sh run.sh

#### Contribution

1.  BioBert SNVs feature extractions, convert language texts to vectors; 
2.  ResNet50 SNVs embedding vectors classification;
3.  PAMPscore mutated genes particibating pathways perturbation score calculation.



