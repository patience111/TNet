import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tqdm
#strategy = tf.distribute.MirroredStrategy()

#load model
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-aels_tall.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-tntypes-ls_tall.h5'))
bac_host = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-host-ls_tall.h5'))
exist_env = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-env-ls_tall.h5'))
asso_args = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-argls_tall.h5'))
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

def encode(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    train_array = np.zeros((1600, 22))
    for i in range(1600):
        if i<len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def test_encode(tests):
    tests_seq = []
    for test in tests:
        tests_seq.append(encode(test))
    tests_seq = np.array(tests_seq)
    
    return tests_seq

def newEncodeVaryLength(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    mol = len(seq) % 16
    dimension1 = len(seq) - mol + 16
    train_array = np.zeros((dimension1,22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    
    return train_array

def test_newEncodeVaryLength(tests):
    tests_seq = []
    for test in tests:
        tests_seq.append(newEncodeVaryLength(test))
    tests_seq = np.array(tests_seq)
    
    return tests_seq

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def filter_prediction_batch(seqs):
    predictions = []
   # for seq in seqs:
    #    temp = model.predict(np.array([seq]))
     #   predictions.append(temp)
    temp = filterm.predict(seqs, batch_size = 512)
    predictions.append(temp)
    return predictions

def prediction(seqs):
    predictions = []
    for seq in seqs:
        temp = model.predict(np.array([seq]))
        predictions.append(temp)
    return predictions

def reconstruction_simi(pres, ori):
    simis = []
    reconstructs = []
    argmax_pre = np.argmax(pres[0], axis=2)
    for index, ele in enumerate(argmax_pre):
        length = len(ori[index])
        count_simi = 0
        #reconstruct = ''
        if length >= 1600:
            align = 1600
        else:
            align = length
        for pos in range(align):
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
#with strategy.scope():
def tnet_lsaa(input_file, outfile):
    #cuts: 0.2373904881101377, 0.23987792946000092, 0.23332709464096324, 0.22396483562213898, 0.23329248366013072
    cut = 0.23357056629867431
    print('reading in test file...')
    test = [i for i in sio.parse(input_file, 'fasta')]

    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
            f.write('test_id' + '\t' + 'inti_type' + '\t' +  'pre_prob' + '\t' + 'bacterial_host' + '\t' + \
                    'pre_prob' + '\t' + 'resistance_category' + '\n')

    print('encoding test file...')
    print('reconstruct, simi...')
    for idx, test_chunk in enumerate(list(chunks(test, 10000))):
    #test_ids = [ele.id for ele in test]
        testencode = test_encode(test_chunk)
        testencode_pre = filter_prediction_batch(testencode) # if huge volumn of seqs (~ millions) this will be change to create batch in advance 
        simis = reconstruction_simi(testencode_pre, test_chunk)
    #results = calErrorRate(simis, cut) 
    #passed = []
        passed_encode = [] ### notice list and np.array
        passed_idx = []
        notpass_idx = []
        for index, ele in enumerate(simis):
            if ele >= cut:
                #passed.append(test[index])
                passed_encode.append(testencode[index])
                passed_idx.append(index)
            else:
                notpass_idx.append(index)
    
        ###classification
        print('classifying...')
        
        if len(passed_encode) > 0:
            classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 512)
            classification_argmax = np.argmax(classifications, axis=1)
            classification_max = np.max(classifications, axis=1)
            
            hosts = bac_host.predict(np.stack(passed_encode, axis=0), batch_size = 512)
            hosts1 = np.squeeze(np.where(hosts >= 0.5, 1, 0))
            if len(hosts1.shape) == 1:
                hosts1 = hosts1.reshape((len(hosts1.shape), len(hosts_prepare)))
            hosts2 = [[x for x, pred in zip(hosts_prepare, y) if pred ==1] for y in hosts1]

            envs = bac_host.predict(np.stack(passed_encode, axis=0), batch_size = 512)
            envs1 = np.squeeze(np.where(envs >= 0.5, 1, 0))
            if len(envs1.shape) == 1:
                envs1 = envs1.reshape((len(envs1.shape), len(envs_prepare)))
            envs2 = [[x for x, pred in zip(envs_prepare, y) if pred ==1] for y in envs1]  
            
            args = asso_args.predict(np.stack(passed_encode, axis=0), batch_size = 512)
            args1 = np.squeeze(np.where(args >= 0.5, 1, 0))
            #args2 = np.array(args_prepare)[args1]
            print('args1.shape: ', args1.shape)
            
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
            
        if len(passed_encode) == 0:
            print('no seq passed!')
#            if idx in notpass_idx:
#                f.write(test_chunk[idx].id + '\t')
#                f.write('not-passed filter' + '\t' + '' + '\t' + '' + '\t' + '' + '\t' + '' + '\n')
            pass

