import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tqdm
import cProfile, pstats, io
import Bio.Data.CodonTable as bdc
from itertools import product
from kito import reduce_keras_model

strategy = tf.distribute.MirroredStrategy()
#model
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-aess_tall.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-tntypes-ss_tall.h5'))
bac_host = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-host-ss_tall.h5'))
exist_env = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-env-ss_tall.h5'))
asso_args = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-argss_tall.h5'))

#encode, encode all the sequence to 1600 aa length
char_dict = {}
chars = 'ACDEFGHIKLMNPQRSTVWXYBJZ'
new_chars = "ACDEFGHIKLMNPQRSTVWXY"
for char in chars:
    temp = np.zeros(22)
    if char == 'B':
        for ch in 'DN':
            temp[new_chars.index(ch)] = 0.5
    elif char == 'J':
        for ch in 'IL':
            temp[new_chars.index(ch)] = 0.5
    elif char == 'Z':
        for ch in 'EQ':
            temp[new_chars.index(ch)] = 0.5
    else:
        temp[new_chars.index(char)] = 1
    char_dict[char] = temp

### encode one test seqs
def newEncodeVaryLength(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    if len(seq) in range(30, 40):
        dimension1 = 32
    if len(seq) in range(40, 50):
        dimension1 = 48
    if len(seq) == 50:
        dimension1 = 64
    train_array = np.zeros((dimension1,22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def test_newEncodeVaryLength(tests):
    tests_seq = [newEncodeVaryLength(test) for test in tests]
    return tests_seq

def encode64(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    dimension1 = 64
    #pad = np.array(21*[0] + [1])
    #time = dimension1-len(seq)
    #train_array = np.stack([char_dict[c] for c in seq]+[pad]*(time))
    train_array = np.zeros((dimension1, 22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def testencode64(seqs):
    encode = [encode64(test) for test in seqs]
    encode = np.array(encode)
    return encode

comprehensive_coden_table = {
    "UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"*", "UAG":"*",
    "UGU":"C", "UGC":"C", "UGA":"*", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
    "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*",
    "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
    "CCT":"P", "CAT":"H", "CGT":"R", "ATT":"I", 
    "ATC":"I", "ATA":"I", "ATG":"M", "ACT":"T",
    "AAT":"N", "AGT":"S", "GTT":"V", "GTC":"V", 
    "GTA":"V", "GTG":"V", "GCT":"A", "GAT":"D",
    "GGT":"G", 'aaa': 'K', 'aac': 'N','aag': 'K',
    'aat': 'N', 'aau': 'N', 'aca': 'T', 'acc': 'T',
    'acg': 'T', 'act': 'T', 'acu': 'T', 'aga': 'R',
    'agc': 'S', 'agg': 'R', 'agt': 'S', 'agu': 'S',
    'ata': 'I', 'atc': 'I', 'atg': 'M', 'att': 'I',
    'aua': 'I', 'auc': 'I', 'aug': 'M', 'auu': 'I',
    'caa': 'Q', 'cac': 'H', 'cag': 'Q', 'cat': 'H',
    'cau': 'H', 'cca': 'P', 'ccc': 'P', 'ccg': 'P',
    'cct': 'P', 'ccu': 'P', 'cga': 'R', 'cgc': 'R',
    'cgg': 'R', 'cgt': 'R', 'cgu': 'R', 'cta': 'L',
    'ctc': 'L', 'ctg': 'L', 'ctt': 'L', 'cua': 'L',
    'cuc': 'L', 'cug': 'L', 'cuu': 'L', 'gaa': 'E',
    'gac': 'D', 'gag': 'E', 'gat': 'D', 'gau': 'D',
    'gca': 'A', 'gcc': 'A', 'gcg': 'A', 'gct': 'A',
    'gcu': 'A', 'gga': 'G', 'ggc': 'G', 'ggg': 'G',
    'ggt': 'G', 'ggu': 'G', 'gta': 'V', 'gtc': 'V',
    'gtg': 'V', 'gtt': 'V', 'gua': 'V', 'guc': 'V',
    'gug': 'V', 'guu': 'V', 'taa': '*', 'tac': 'Y',
    'tag': '*', 'tat': 'Y', 'tca': 'S', 'tcc': 'S',
    'tcg': 'S', 'tct': 'S', 'tga': '*', 'tgc': 'C',
    'tgg': 'W', 'tgt': 'C', 'tta': 'L', 'ttc': 'F',
    'ttg': 'L', 'ttt': 'F', 'uaa': '*', 'uac': 'Y',
    'uag': '*', 'uau': 'Y', 'uca': 'S', 'ucc': 'S',
    'ucg': 'S', 'ucu': 'S', 'uga': '*', 'ugc': 'C',
    'ugg': 'W', 'ugu': 'C',  'uua': 'L', 'uuc': 'F',
    'uug': 'L', 'uuu': 'F'}

ambiguous = ["A","C","G","T","U","W","S","M","K","R","Y","B","D","H","V","N"]
keywords = [''.join(i) for i in product(ambiguous, repeat = 3)]
keywords_select = [ele for ele in keywords if ele not in comprehensive_coden_table.keys()]
keywords_select_dict = {ele : 'X' for ele in keywords_select}
keywords_select_lower_dict = {ele.lower() : 'X' for ele in keywords_select}

finalT = {}
finalT.update(comprehensive_coden_table)
finalT.update(keywords_select_dict)
finalT.update(keywords_select_lower_dict)

def translate(seq):
    lenseq = len(seq)
    aa = ['*']*(lenseq//3)
    for i in range(0, lenseq-lenseq%3, 3):
        codon = seq[i:i+3]
    #if codon in forwardT:
        aa[i//3] = finalT[codon]
    aastr = ''.join(aa)
    return aastr

def test_encode(seqs):
    """
    input as a list of test sequences
    """
    record_notpre = []
    record_notpre_idx = []
    record_pre = {}
    encodeall_dict = {}
    encodeall = []
    start = 0
    ori = []
    #length = length
    for idx, seq in tqdm.tqdm(enumerate(seqs)):
        #/print(seq.id)
        seqf = str(seq.seq)
        rc = str(seq.seq.reverse_complement())
        temp = [translate(seqf), translate(seqf[1:]), translate(seqf[2:]), translate(rc), translate(rc[1:]), translate(rc[2:])]
        #temp = [seq.seq.translate(), seq.seq[1:].translate(), seq.seq[2:].translate(), rc.translate(), rc[1:].translate(), rc[2:].translate()]
        temp_split = []
        for ele in temp:
            if "*" in ele:
                temp_split.extend(ele.split('*'))
            else:
                temp_split.append(ele)
        temp_seq = [str(ele) for index, ele in enumerate(temp_split) if len(ele) >= 25]
        #print(len(temp_seq))

        if len(temp_seq) == 0:
            record_notpre.append(seq.id)
            continue
        else:
            record_pre[seq.id] = idx
            ori.extend(temp_seq)
            encode = testencode64(temp_seq)
            encodeall_dict[seq.id] = (start, start + len(temp_seq))
            encodeall.extend(encode)
            start += len(temp_seq)
    encodeall = np.array(encodeall)
    return encodeall, record_notpre, record_notpre_idx, record_pre, encodeall_dict, ori

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def prediction(seqs):
    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)
    return predictions

def reconstruction_simi(pres, ori):
    simis = []
    reconstructs = []
    argmax_pre = np.argmax(pres, axis=2)
    for index, ele in enumerate(argmax_pre):
        length = len(ori[index])
        count_simi = 0
        #reconstruct = ''
        for pos in range(length):
            if chars[ele[pos]] == ori[index][pos]:
                count_simi += 1
            #reconstruct += chars[np.argmax(ele[pos])]
        simis.append(count_simi / length)
        #reconstructs.append(reconstruct)
    return simis

tns_labels = ['Tn554_tnpA', 'Tn1071_tnpA', 'Tn554_tnpB', 'Tn4430_tnpA', 'Tn4651_tnpA', 'Tn3000_tnpA',
              'Tn163_tnpA', 'Tn7_tniA', 'Tn3_tnpA', 'Tn21_tnpA']

hosts_labels = ['Acidithiobacillia', 'Acidobacteriia', 'Actinomycetia', 'Alphaproteobacteria', 'Bacilli',
 'Betaproteobacteria', 'Clostridia', 'Cytophagia', 'Deinococci', 'Deltaproteobacteria', 'Epsilonproteobacteria',
 'Flavobacteriia', 'Fusobacteriia', 'Gammaproteobacteria', 'Hydrogenophilalia', 'Negativicutes',
 'Nitrospira', 'Spartobacteria', 'Thermomicrobia', 'Tissierellia', 'Unclassified']

envs_labels = ['Animal', 'Freshwater', 'Human', 'Other_biotic', 'Other_environment', 'Plant', 'Seawater', 
        'Soil_sediment', 'Unknown', 'Wastewater']

args_labels = ['MLS', 'Aminocoumarin', 'Aminoglycoside', 'Bacitracin', 'Beta-lactam', 'Bleomycin', 
'Chloramphenicol', 'Elfamycin', 'Ethambutol', 'Fosfomycin', 'Fosmidomycin', 'Fusidic_acid', 
'Glycopeptide', 'Isoniazid', 'Kasugamycin', 'Multidrug', 'Mupirocin', 'Nitrofurantoin', 'Nitroimidazole', 
'Oxazolidinone', 'Peptide', 'Pleuromutilin', 'Polymyxin', 'Pyrazinamide', 'Qa_compound', 'Quinolone',
'Rifamycin', 'Streptothricin', 'Sulfonamide', 'Tetracenomycin', 'Tetracycline', 'Triclosan', 
'Trimethoprim', 'Tunicamycin', 'Unknown']

tns_prepare = sorted(tns_labels)
tns_dic = {}
for index, ele in enumerate(tns_prepare):
    tns_dic[index] = ele

hosts_prepare = sorted(hosts_labels)

envs_prepare = sorted(envs_labels)

args_prepare = sorted(args_labels)

#cuts = [0.6, 0.6141439205955335, 0.5913729128014842] #AESS_tns_nt_results
cut = 0.6018389444656725

#@profile
with strategy.scope():
    def tnet_ssnt(input_file, outfile):
        with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
            f.write('test_id' + '\t' + 'tn_type' + '\t' +  'pre_prob' + '\t' + 'bacterial_host'  + '\t' + \
                    'exsiting environment' + '\t' + 'resistance_category' + '\n') 
        
        #testencode_pre = []
        print('reading in test file...')
        test = [i for i in sio.parse(input_file, 'fasta')]
        #test_ids = [ele.id for ele in test]
        #arg_encode, record_notpre, record_pre, encodeall_dict, ori = test_encode(arg, i[-1])
        print('encoding test file...')
        for idx, test_chunk in enumerate(list(chunks(test, 10000))):
            testencode_pre = []
            testencode, not_pre, not_pre_idx, pre, encodeall_dict, ori  = test_encode(test_chunk)
            #print('len ori: ', len(ori))

            for num in range(0, len(testencode), 8192):
                testencode_pre += prediction(testencode[num:num+8192])
            #testencode_pre = prediction(testencode) # if huge volumn of seqs (~ millions) this will be change to create batch in advance 
            pre_con = np.concatenate(testencode_pre)
            simis = reconstruction_simi(pre_con, ori)
            passed_encode = [] ### notice list and np.array
            passed_idx = []
            notpass_idx = []
            assert len(simis) == len(ori)
            simis_edit = []
            count_iter = 0

            for k, v in encodeall_dict.items():
                simis_edit.append(max(simis[v[0]:v[-1]]))
                count_iter += 1
            for index, ele in enumerate(simis_edit):
        #        if len(test[index]) < 120:
        #            cuts_idx = 0
        #        elif len(test[index]) < 150:
        #            cuts_idx = 1
        #        else:
        #            cuts_idx = 2
        #        if ele >= cuts[cuts_idx]:
        #            passed_encode.append(testencode[index])
        #            passed_idx.append(index)
        #        else:
        #            notpass_idx.append(index)
                if ele >= cut:
                    passed_encode.append(testencode[index])
                    passed_idx.append(index)
                else:
                    notpass_idx.append(index) 
            
            ###classification
            print('classifying...')
            if len(passed_encode) > 0:
                classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
                classification_argmax = np.argmax(classifications, axis=1)
                classification_max = np.max(classifications, axis=1)

                hosts = bac_host.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
                hosts1 = np.squeeze(np.where(hosts >= 0.5, 1, 0))
                if len(hosts1.shape) == 1:
                    hosts1 = hosts1.reshape((len(hosts1.shape), len(hosts_prepare)))
                hosts2 = [[x for x, pred in zip(hosts_prepare, y) if pred ==1] for y in hosts1]

                envs = exist_env.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
                envs1 = np.squeeze(np.where(envs >= 0.5, 1, 0))
                if len(envs1.shape) == 1:
                    envs1 = envs1.reshape((len(envs1.shape), len(envs_prepare)))
                envs2 = [[x for x, pred in zip(envs_prepare, y) if pred ==1] for y in envs1]

                args = asso_args.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
                args1 = np.squeeze(np.where(args >= 0.5, 1, 0))
                if len(args1.shape) == 1:
                    args1 = args1.reshape((len(args1.shape), len(args_prepare)))
                args2 = [[x for x, pred in zip(args_prepare, y) if pred ==1] for y in args1]
                assert len(args2)==len(passed_idx)
                
                tn = {}
                hosts2_e1 = {}
                envs2_e1 = {}
                args2_e1 = {}
                for i, ele in enumerate(passed_idx):
                    tn[ele] = [classification_max[i], tns_dic[classification_argmax[i]]]
                    hosts2_e1[ele] = hosts2[i]
                    envs2_e1[ele] = envs2[i]
                    args2_e1[ele] = args2[i]
            ### output
                print('writing output...')
                with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
                    for idx, ele in enumerate(test_chunk):
                        if idx in passed_idx:
                            f.write(test_chunk[idx].id + '\t')
                            f.write(tn[idx][-1] + '\t')
                            f.write(str(tn[idx][0]) + '\t')
                            f.write(','.join(hosts2_e1[idx]) + '\t')
                            f.write(','.join(envs2_e1[idx]) + '\t')
                            f.write(','.join(args2_e1[idx]) + '\n')

                        if idx in notpass_idx:
                            f.write(test_chunk[idx].id + '\t')
                            f.write('non-Tn' + '\t' + '' + '\t' + '' + '\t' + '' + '\t' + '' + '\n')
                        if idx in not_pre_idx:
                            f.write(test_chunk[idx].id + '\t')
                            f.write('<25aa' + '\t' + '' + '\t' + '' + '\t' + '' + '\t' + '' + '\n')
            
            if len(passed_encode) == 0:
                print('no seq passed!')
                pass
        
