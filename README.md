TNet
=====
Multi-task multilabel deep neural networks for identification and classification of transposons.

TNet is designed to identify and classify transposon transposase, predict bacterial hosts (multi-label), environmental context(multi-label), and associate ARGs (multi-label) with transposons. This versatile tool supports a range of input types including:
* Long Amino Acid Sequences (Full Length/Contigs)
* Long Nucleotide Sequences
* Short Amino Acid Reads (30-50 aa)
* Short Nucleotide Reads (100-150 nt)

All inputs should be in FASTA format.

**TNet Components**\
TNet comprises two specialized models to accommodate different read lengths:
* **TNet-s**: Optimized for short reads, enhancing prediction accuracy for sequences ranging from 30 to 50 amino acids or 100 to 150 nucleotides.
* **TNet-l**: Tailored for long sequences, ensuring robust predictions for full-length contigs or long nucleotide sequences.

![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_workflow.jpg)

Installation
------------
clone the program to your local machine\
git clone https://github.com/patience111/TNet.git


**1. Setting up environment**


**1.1 Installation with conda**


1.1.1 For **CPU** inference, you could install the program with conda YAML file in the installation directory with the following commands:

```
cd ./installation 
conda env create -f TNet-CPU.yml -n TNet-cpu
conda activate TNet-cpu
```

(This was tested on Ubuntu 16.04, 20.04; Windows 10, macOS(14.1.1))\
 ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet-cpu_test_e1.jpg)

  1.1.2 For **GPU** inference, you could install the program with conda YAML file in the installation directory with the following commands:</br>
```
cd ./installation
conda env create -f TNet-GPU.yml -n TNet-gpu
conda activate TNet-gpu
```
(This was tested on Ubuntu 16.04, cuda 10.1, Driver Version: 430.64)\
    ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet-gpu_test_p1.jpg)
    ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet-gpu_test_p2.jpg)
    
    

