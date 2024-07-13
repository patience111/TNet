import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm

filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/94_0.4_AESS_dim22_f42_swor_pad48_strip_e2000_round_add3mreslast.best.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-tntypes-ss_tall.h5'))
bac_host = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-host-ss_tall.h5'))
asso_args = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/tnet-argss_tall.h5'))

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
    train_array = np.zeros((dimension1,22))
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def prediction(seqs):
    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)
    return predictions

def filter_prediction_batch(seqs):

    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)

    return predictions

def reconstruction_simi(pres, ori):
    simis = []
    #reconstructs = []
    argmax_pre = np.argmax(pres[0], axis=2)
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

intis_labels = ['Group09', 'Group10', 'Group13', 'Group03', 'Group02', 'Group01', 'Group05', 'Group11',
 'Group06', 'Unclassified', 'Group12', 'Group04', 'Group07', 'Group14', 'Group15', 'Group08']

hosts_labels = ['Gammaproteobacteria',  'Unassigned',  'Betaproteobacteria',  'Deltaproteobacteria',  
           'Planctomycetia',  'Spirochaetia',  'Anaerolineae',  'Verrucomicrobiae',  'Opitutae',  
           'Epsilonproteobacteria',  'Ignavibacteria',  'Gemmatimonadetes',  'Candidatus Brocadiia',  
           'Balneolia',  'Spartobacteria',  'Kiritimatiellae',  'Blastocatellia',  'Chlorobia',  'Nitrospira',
           'Actinomycetia',  'Hydrogenophilalia',  'Phycisphaerae',  'Chloroflexia',  'Alphaproteobacteria',  
           'Holophagae',  'Acidithiobacillia',  'Zetaproteobacteria',  'Coriobacteriia',  'Lentisphaeria',  
           'Rhodothermia',  'Caldilineae',  'Bacteroidia',  'Acidobacteriia',  'Clostridia',  'Calditrichia', 
           'Thermoanaerobaculia',  'Thermodesulfobacteria',  'Vicinamibacteria',  'Deinococci',  'Bacilli',  
           'Chitinophagia',  'Ardenticatenia',  'Thermoflexia',  'Acidimicrobiia',  'Flavobacteriia',  
           'Tichowtungiia',  'Thermodesulfovibrionia',  'Fimbriimonadia',  'Chitinivibrionia',  'Oligoflexia', 
           'Candidatus Fermentibacteria (class)',  'Candidatus Lambdaproteobacteria',  
           'Candidatus Muproteobacteria',  'Abditibacteria',  'Candidatus Binatia',  'Limnochordia',  
           'Synergistia',  'Candidatus Thermofonsia',  'Thermoleophilia',  'Cytophagia',  'Tepidiformia',  
           'Aquificae',  'Chrysiogenetes',  'Gloeobacteria',  'Fibrobacteria',  'Chitinispirillia',  
           'Dehalococcoidia',  'Candidatus Ozemobacteria',  'Candidatus Polarisedimenticolia',  
           'Erysipelotrichia']

args_labels = ['MLS', 'aminocoumarin', 'aminoglycoside', 'bacitracin', 'beta-lactam', 'bleomycin', 'chloramphenicol', 'elfamycin', 'ethambutol', 'fosfomycin', 'fosmidomycin', 'fusidic_acid', 'glycopeptide', 'isoniazid', 'kasugamycin', 'multidrug', 'mupirocin', 'nitrofurantoin', 'nitroimidazole', 'peptide', 'pleuromutilin', 'polymyxin', 'pyrazinamide', 'qa_compound', 'quinolone', 'rifamycin', 'streptothricin', 'sulfonamide', 'tetracenomycin', 'tetracycline', 'triclosan', 'trimethoprim', 'tunicamycin']

intis_prepare = sorted(intis_labels)
intis_dic = {}
for index, ele in enumerate(intis_prepare):
    intis_dic[index] = ele


hosts_prepare = sorted(hosts_labels)
hosts_dic = {}
for index, ele in enumerate(hosts_prepare):
    hosts_dic[index] = ele


args_prepare = sorted(args_labels)

#cuts = [0.7666666666666667, 0.8, 0.8] ls126, rate0.6
cut = 0.7888888888888891

def tnet_ssaa(input_file, outfile):

    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
            f.write('test_id' + '\t' + 'inti_type' + '\t' +  'pre_prob' + '\t' + 'bacterial_host' + '\t' + \
                    'pre_prob' + '\t' + 'resistance_category' + '\n')
    print('reading in test file...')
    test = [i for i in sio.parse(input_file, 'fasta')]

    print('encoding test file...')
    print('reconstruct, simi...')
    for idx, test_chunk in enumerate(list(chunks(test, 10000))):
        #print(str(idx) + 'th batch encoding...')
        testencode = testencode64(test_chunk)
        #print(str(idx) + 'th batch predict...')
        testencode_pre = filter_prediction_batch(testencode)
        #print(str(idx) + 'th reconstruct, simi...')
        simis = reconstruction_simi(testencode_pre, test_chunk)
        passed_encode = [] ### notice list and np.array
        passed_idx = []
        notpass_idx = []
        for index, ele in enumerate(simis):
#            if ele >= cuts[0]:
#                passed_encode.append(testencode[index])
#                passed_idx.append(index)
#            else:
#                notpass_idx.append(index)
#            if len(test[index]) in range(40, 50):
#                if ele >= cuts[1]:
#                    passed_encode.append(testencode[index])
#                    passed_idx.append(index)
#                else:
#                    notpass_idx.append(index)
#            if len(test[index]) == 50:
#                if ele >= cuts[-1]:
#                    passed_encode.append(testencode[index])
#                    passed_idx.append(index)
#                else:
#                    notpass_idx.append(index)
            if ele >= cut:
                passed_encode.append(testencode[index])
                passed_idx.append(index)
            else:
                notpass_idx.append(index)

    ###classification
        print('classifying...')
        if len(passed_encode) > 0:
            classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
            #out = {}
            classification_argmax = np.argmax(classifications, axis=1)
            classification_max = np.max(classifications, axis=1)

            hosts = bac_host.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
            hosts_argmax = np.argmax(hosts, axis=1)
            hosts_max = np.max(hosts, axis=1)

            args = asso_args.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
            args1 = np.squeeze(np.where(args >= 0.5, 1, 0))
            #args2 = np.array(args_prepare)[args1]
            args2 = [[x for x, pred in zip(args_prepare, y) if pred ==1] for y in args1]
            assert len(args2)==len(passed_idx)
            
            inti = {}
            host = {}
            args2_e1 = {}

            for i, ele in enumerate(passed_idx):
                inti[ele] = [classification_max[i], intis_dic[classification_argmax[i]]]
                host[ele] = [hosts_max[i], hosts_dic[hosts_argmax[i]]]
                args2_e1[ele] = args2[i]

            ### output
            print('writing output...')
            with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
                for idx, ele in enumerate(test_chunk):
                    if idx in passed_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write(inti[idx][-1] + '\t')
                        f.write(str(inti[idx][0]) + '\t')
                        f.write(host[idx][-1] + '\t')
                        f.write(str(host[idx][0]) + '\t')
                        f.write(','.join(args2_e1[idx]) + '\n')
                    
                    if idx in notpass_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('non-inti' + '\t' + '' + '\t' + '' + '\t' + '' + '\t' + '' + '\n')


        if len(passed_encode) == 0:
            print('no seq passed!')
            pass
