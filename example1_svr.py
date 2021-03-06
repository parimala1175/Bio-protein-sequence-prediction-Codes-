first_protein ="TIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFAGDGLFTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPST"
print(len(first_protein))
second_protein = "KETSPIPQPKTFGPLGNLPLIDKDKPTLSLIKLAEEQGPIFQIHTPAGTTIVVSGHELVKEVCDEERFDKSIEGALEKVRAFSGDGLFTSWTHEPNWRKAHNILMPTFSQRAMKDYHEKMVDIAVQLIQKWARLNPNEAVDVPGDMTRLTLDTIGLCGFNYRFNSYYRETPHPFINSMVRALDEAMHQMQRLDVQDKLMVRTKRQFRYDIQTMFSLVDSIIAERRANGDQDEKDLLARMLNVEDPETGEKLDDENIRFQIITFLIAGHETTSGLLSFATYFLLKHPDKLKKAYEEVDRVLTDAAPTYKQVLELTYIRMILNESLRLWPTAPAFSLYPKEDTVIGGKFPITTNDRISVLIPQLHRDRDAWGKDAEEFRPERFEHQDQVPHHAYKPFGNGQRACIGMQFALHEATLVLGMILKYFTLIDHENYELDIKQTLTLKPGDFHISVQSRHQEAIHADVQAAE"
third_protein = "KQASAIPQPKTYGPLKNLPHLEKEQLSQSLWRIADELGPIFRFDFPGVSSVFVSGHNLVAEVCDEKRFDKNLGKGLQKVREFGGDGLFTSWTHEPNWQKAHRILLPSFSQKAMKGYHSMMLDIATQLIQKWSRLNPNEEIDVADDMTRLTLDTIGLCGFNYRFNSFYRDSQHPFITSMLRALKEAMNQSKRLGLQDKMMVKTKLQFQKDIEVMNSLVDRMIAERKANPDENIKDLLSLMLYAKDPVTGETLDDENIRYQIITFLIAGHETTSGLLSFAIYCLLTHPEKLKKAQEEADRVLTDDTPEYKQIQQLKYIRMVLNETLRLYPTAPAFSLYAKEDTVLGGEYPISKGQPVTVLIPKLHRDQNAWGPDAEDFRPERFEDPSSIPHHAYKPFGNGQRACIGMQFALQEATMVLGLVLKHFELINHTGYELKIKEALTIKPDDFKITVKPRKTAAINVQRKEQA"
print(len(second_protein))
print(len(third_protein))

############# numerical encoding for three proteins #######
dictionary = {'A':0.86,'R':0.94,'N':0.74 ,'D':0.72 ,'C':1.17,'Q':0.89,'E':0.62,'G':0.97, 'H':1.06, 'I':1.24, 'L':0.98, 'Q':0.89 ,  'E':0.62 , 'G': 0.97 ,'H': 1.06 , 'I':  1.24 ,'L' :0.98,  'K' :0.79,  'M' :1.08,  'F' :1.16, 'P':1.22,'S' :1.04,'T':1.18, 'W' :1.07,'Y' :1.25, 'V':1.33}
#################
import numpy as np
first_protein_array = np.zeros(463)
second_protein_array = np.zeros(466)
third_protein_array = np.zeros(466)
first = list(first_protein)
for i in range(len(first_protein)):
    first_protein_array[i]=dictionary.get(first[i])    
#print(first_protein_array)
second = list(second_protein)
for i in range(len(second_protein)):
    second_protein_array[i]=dictionary.get(second[i])
third = list(third_protein)
for i in range(len(third_protein)):
    third_protein_array[i]=dictionary.get(third[i])
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.debugger import Pdb
pdb = Pdb()
from scipy.signal import savgol_filter
from sklearn import preprocessing,svm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
#data = pd.read_excel('Data for python analysis.xlsx')
#### 
X_calib = np.zeros([196,3728])
Y_calib = np.zeros(196)
text_file = open("InnovSAR.txt", "r")
#lines = text_file.readlines()
lines = text_file.read().split('\n')
for j in range(195):
    string = lines[j].split(' ')
    #print(string)
    ff = [int(i) for i in string[0]]
    #X_calib[j,:] = ff
    Y_calib[j] = float(string[1])
    x=0
    for k in range(8):
        if(ff[k] == 1):
            X_calib[j,x:x+463] = first_protein_array
            x=x+463
        elif(ff[k] == 2):
            X_calib[j,x:x+466] = second_protein_array
            x=x+466
        else:
            X_calib[j,x:x+466] = third_protein_array
            x=x+466
    #X_fft = np.abs(np.fft.fft(X_calib[j,:]))
    #X_calib[j,:] = X_fft
    #pdb.set_trace()
#print(type(lines[0]))
#print lines
#print len(lines)
#text_file.close()
#
#print(X_calib)
#pdb.set_trace()
# Get reference values
#reference_data = pd.DataFrame.as_matrix(data['Ref AC'])
#Y_calib = reference_data[:422]
#Y_valid = reference_data[423:]
#print X_train.shape
#print Y_train.shape
X_valid = np.zeros([47,3728])
Y_valid = np.zeros(47)
text_file = open("validation_SAR.txt", "r")
#lines = text_file.readlines()
lines = text_file.read().split('\n')
for j in range(47):
    string = lines[j].split(' ')
    #print(string)
    ff = [int(i) for i in string[0]]
    #X_valid[j,:] = ff
    Y_valid[j] = float(string[1])
    x=0
    for k in range(8):
        if(ff[k] == 1):
            X_valid[j,x:x+463] = first_protein_array
            x=x+463
        elif(ff[k] == 2):
            X_valid[j,x:x+466] = second_protein_array
            x=x+466
        else:
            X_valid[j,x:x+466] = third_protein_array
            x=x+466
    #X_fft = np.abs(np.fft.fft(X_valid[j,:]))
    #X_valid[j,:] = X_fft
#pdb.set_trace()
# Get spectra
#X_calib = 
#X_valid = 
##################### converting to numerical encoding ############

############### spectrum calculation ################
# Get wavelengths (They are in the first line which is considered a header from pandas)
#wl = np.array(list(data)[2:])
    
# Plot spectra
#plt.figure(figsize=(8,4.5))
#with plt.style.context(('ggplot')):
    #plt.plot(wl, X_calib.T)
    #plt.xlabel('Wavelength (nm)')
    #plt.ylabel('Absorbance')    
#plt.show()
# Calculate derivatives
#X2_calib = savgol_filter(X_calib, 17, polyorder = 2,deriv=2)
#X2_valid = savgol_filter(X_valid, 17, polyorder = 2,deriv=2)

# Plot second derivative
#plt.figure(figsize=(8,4.5))
#with plt.style.context(('ggplot')):
 #   plt.plot(wl, X2_calib.T)
 #   plt.xlabel('Wavelength (nm)')
 #   plt.ylabel('D2 Absorbance')
#plt.show()
clf = svm.SVR(kernel='rbf')
clf.fit(X_calib, Y_calib)
confidence = clf.score(X_valid, Y_valid)
print(confidence)


