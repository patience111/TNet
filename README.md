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

![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_graphic_abstract-logo_250328.png)

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
 ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet-cpu_test_e2.jpg)

  1.1.2 For **GPU** inference, you could install the program with conda YAML file in the installation directory with the following commands:</br>
```
cd ./installation
conda env create -f TNet-GPU.yml -n TNet-gpu
conda activate TNet-gpu
```
(This was tested on Ubuntu 16.04, cuda 10.1, Driver Version: 430.64)\
    ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_gpu-test_p1.jpg)
    ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_gpu-test_p2.jpg)


**1.2 Or, if you prefer installing dependencies manually, you might find this information useful:**\
The program was tested with the following package version, you can install exactly the same version or other compatible versions.
```
Biopython:  1.79
tensorflow:  2.2.0 
cuda: 10.2 (for GPU using)
cudnn: 7.6.5.32 (for GPU using)
numpy: 1.18.5
scikit-learn: 0.24.1
tqdm: 4.56.0
```
**2. Getting trained models**

```   
cd ./models
bash get-models.sh
```
Quickstart Guide
----------------
***for long sequences***

```
python tnet.py --input input_path_data  --type aa/nt --model tnet-l  --outname output_file_name
```
***for short reads***

```
python tnet.py --input input_path_data --type aa/nt --model tnet-s --outname output_file_name
```

**general options:**</br>
     --input/-i&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the test file as input </br>
     --type/-t &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;molecular type of your test data (aa for amino acid, nt for nucleotide)</br>
     --model/-m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the model you assign to make the prediction (tnet-l for long sequences, tnet-s for short reads) </br>
     --outname/-on&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the output file name </br>


**optional arguments:**</br>
  -h, --help            show this help message and exit</br></br>
  ![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_help-page_e2.jpg)</br>
  -i INPUT, --input INPUT </br>
                        the test data as input </br></br>
  -t {aa,nt}, --type {aa,nt} </br>
                        molecular type of your input file </br></br>
  -m {tnet-s,tnet-l}, --model {tnet-s,tnet-l} </br>
                        the model to make the prediction </br></br>
  -on OUTNAME, --outname OUTNAME </br>
                        the name of results output </br></br>


Example
----------
if we predict the long amino acid sequences by using TNet-l model, we could use command line (if you are in TNet dirctory):
```
python3 ./scripts/tnet.py --input ../Tests/tn3015_100p.fasta --type aa --model tnet-l --outname TNet_test-gpu.txt
```
**output** will be like and saved in the **results** folder: </br>
![alt text](https://github.com/patience111/TNet/blob/master/pics/TNet_gpuTest_lsaa.jpg)</br>
The first column **test_id** is the sequence label of the test sequnece.</br>
The second column **tnp_type** is the "unit transposon" or "non-unit transposon" prediction of the input sequence.\
The third column **pre_prob** is the unit transposon prediction confidence of the input sequence by the model.\
The fourth column **bacterial_host** is the multi-label bacterial host prediction of the input sequence if it is predicted as unit transposon first.\
The fifth column **env_context** is the multi-label prediction of environmental context of the input sequence if it is predicted as unit transposon first.\
The last column **resistance_category** is the multi-label prediction of asssociated ARGs of the input sequences.

Contribute
----------

If you'd like to contribute to TNet, check out https://github.com/patience111/TNet. \
Hope you enjoy INTNet journey, any problem please contact scpeiyao@gmail.com  
    

