"""Main module."""
import argparse
import textwrap
#import tnet_lsaa as lsaa
#import tnet_lsnt as lsnt
#import tnet_ssaa as ssaa
#import tnet_ssnt as ssnt
import sys
parser = argparse.ArgumentParser(
prog='TNet',
formatter_class=argparse.RawDescriptionHelpFormatter,
description=textwrap.dedent("""\
   TNet: Multitask and multilabel deep neural network for identication and classification bacterial transposon.
   --------------------------------------------------------------------------------------------------------
    The standlone program is at https://github.com/id-bioinfo/IntNet and  https://github.com/patience111/IntNet.
    The online service will be released later.

    The input can be long amino acid sequences(full length/contigs), long nucleotide sequences,
    short amino acid reads (30-50aa), short nucleotide reads (100-150nt) in fasta format.
    If your input is short reads you should assign 'tnet-s' model, or if your input is full-length/contigs
    you should assign 'tnet-l' to make the predict.

    The output of the program is a table contain integron group, bacteria host, and associated ARGs(multilabel)
    inferred by the deep learning models.

    USAGE:
        for full-length or contigs
            python tnet.py --input input_path_data --type aa/nt --model tnet-l  --outname output_file_name
        for short reads
            python tnet.py --input input_path_data --type aa/nt --model tnet-s  --outname output_file_name

    general options:
        --input/-i    the test file as input
        --type/-t     molecular type of your test data (aa for amino acid, nt for nucleotide)
        --model/-m    the model you assign to make the prediction (tnet-l for long sequences, tnet-s for short reads)
        --outname/-on  the output file name
    """

),
epilog='Hope you enjoy TNet journey, any problem please contact scpeiyao@gmail.com\n')

parser.print_help()

parser.add_argument('-i', '--input', required=True, help='the test data as input')
parser.add_argument('-t', '--type', required=True, choices=['aa', 'nt'], help='molecular type of your input file')
parser.add_argument('-m', '--model', required=True, choices=['tnet-s', 'tnet-l'], help='the model to make the prediction')
parser.add_argument('-on', '--outname', required=True, help='the name of results output')

args = parser.parse_args()

## for AESS_aa -> classifier
if args.type == 'aa' and args.model == 'tnet-s':
    import tnet_ssaa as ssaa
    ssaa.tnet_ssaa(args.input, args.outname)

# for AESS_nt -> classifier
if args.type == 'nt' and args.model == 'tnet-s':
    import  tnet_ssnt_g3 as ssnt
    ssnt.tnet_ssnt(args.input, args.outname)

# for AELS_aa -> classifier
if args.type == 'aa' and args.model == 'tnet-l':
    import tnet_lsaa_g3 as lsaa
    lsaa.tnet_lsaa(args.input, args.outname)

# for AELS_nt -> classifier
if args.type == 'nt' and args.model == 'tnet-l':
    import tnet_lsn_g3 as lsnt
    lsnt.tnet_lsnt(args.input, args.outname)
