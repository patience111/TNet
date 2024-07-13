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

![alt text] (https://github.com/patience111/TNet/blob/master/pics/TNet_workflow.jpg)
