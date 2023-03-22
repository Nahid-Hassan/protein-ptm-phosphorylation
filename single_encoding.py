#import re
#from collections import Counter
from descproteins import *
from pubscripts import save_file

tags = ["Train"]
residues = ["#"]
for residue in residues:
    for tag in tags:
        #fname = "dataset/"+tag+"W21"+residue+".txt"
        #fname = "/home/nahid/Desktop/iLearn_utpal/ptm-final-dataset.csv"
        #fname = "C:/Users/tc/Desktop/4th-year-project-final/iLearn_AbirModifiedFinal/pp.csv"
        #fname = "./ptm-final-dataset.csv"
        fname = "./pompompaac.csv"
        fastas = []
        with open(fname, 'r') as fin:
            for line in fin:
                words = line.split(",")

                name = words[1]

                sequence = words[0]

                label = words[2].strip()

                fastas.append([name, sequence, label])
            myOrder = "ACDEFGHIKLMNPQRSTVWY"
            filePath = "data/"
            kw = {'path': filePath, 'order': myOrder, 'type': 'Protein'}
            choices = ['PAAC']
            # choices = ['PSSM', 'SSEC', 'SSEB']
            #choices = ['AAC', 'EAAC', "TPC",'DDE', 'CKSAAP', 'binary','GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC','AAINDEX','ZSCALE', 'BLOSUM62','CTDC', 'CTDT', 'CTDD']
                #   'DPC', 'APAAC', 'PAAC','NMBroto', 'Moran', 'Geary','CTriad', 'KSCTriad', 'KNNprotein','KNNpeptide',
                #   'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA']
                #choices = ['AAC', 'EAAC', 'CKSAAP', 'DPC', 'DDE', 'TPC', 'binary',
                    #'GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC',
                    #'AAINDEX', 'ZSCALE', 'BLOSUM62',
                    #'NMBroto', 'Moran', 'Geary',
                    #'CTDC', 'CTDT', 'CTDD',
                    #'CTriad', 'KSCTriad',
                    #'SOCNumber', 'QSOrder',
                    #'PAAC', 'APAAC',
                    #'KNNprotein', 'KNNpeptide',
                    #'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA'
                    #]
            for desc in choices:
                cmd = desc + '.' + desc + '(fastas, **kw)'

                encodings = eval(cmd)
                #print(encodings[:]) 
                #fout = "Encoding_result/"+ "single_encoding_ptm" +".txt"
                #save_file.save_file(encodings, "csv", fout)
                fout = "Encoding_result/"+ "paacFeatureEncoding" +".txt"
                save_file.save_file(encodings, "csv", fout)

            
