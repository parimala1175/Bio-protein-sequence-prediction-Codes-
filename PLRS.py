############################# Protein structure ##########
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.debugger import Pdb
pdb = Pdb()
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
#data = pd.read_excel('Data for python analysis.xlsx')
#### 
X_calib = np.zeros([196,8])
Y_calib = np.zeros(196)
text_file = open("InnovSAR.txt", "r")
#lines = text_file.readlines()
lines = text_file.read().split('\n')
for j in range(195):
    string = lines[j].split(' ')
    #print(string)
    ff = [int(i) for i in string[0]]
    X_calib[j,:] = np.abs(np.fft.fft(ff))
    Y_calib[j] = float(string[1])
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
X_valid = np.zeros([47,8])
Y_valid = np.zeros(47)
text_file = open("validation_SAR.txt", "r")
#lines = text_file.readlines()
lines = text_file.read().split('\n')
for j in range(47):
    string = lines[j].split(' ')
    #print(string)
    ff = [int(i) for i in string[0]]
    X_valid[j,:] = ff
    Y_valid[j] = float(string[1])
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
def prediction(X_calib, Y_calib, X_valid, Y_valid, plot_components=False):

    #Run PLS including a variable number of components, up to 40,  and calculate MSE
    mse = []
    component = np.arange(1,8)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Fit
        pls.fit(X_calib, Y_calib)
        # Prediction
        Y_pred = pls.predict(X_valid)

        mse_p = mean_squared_error(Y_valid, Y_pred)
        mse.append(mse_p)

        comp = 100*(i+1)/8
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(xmin=-1)

        plt.show()

    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=msemin+1)
    pls.fit(X_calib, Y_calib)
    Y_pred = pls.predict(X_valid) 
    
    # Calculate and print scores
    score_p = r2_score(Y_valid, Y_pred)
    mse_p = mean_squared_error(Y_valid, Y_pred)
    sep = np.std(Y_pred[:,0]-Y_valid)
    rpd = np.std(Y_valid)/sep
    bias = np.mean(Y_pred[:,0]-Y_valid)
    
    print('R2: %5.3f'  % score_p)
    print('MSE: %5.3f' % mse_p)
    print('SEP: %5.3f' % sep)
    print('RPD: %5.3f' % rpd)
    print('Bias: %5.3f' %  bias)

    # Plot regression and figures of merit
    rangey = max(Y_valid) - min(Y_valid)
    rangex = max(Y_pred) - min(Y_pred)

    z = np.polyfit(Y_valid, Y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(Y_pred, Y_valid, c='red', edgecolors='k')
        ax.plot(z[1]+z[0]*Y_valid, Y_valid, c='blue', linewidth=1)
        #ax.plot(Y_valid, Y_valid, color='green', linewidth=1)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.title('Prediction')

        # Print the scores on the plot
        plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.1*rangey, 'R$^{2}=$ %5.3f'  % score_p)
        plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.15*rangey, 'MSE: %5.3f' % mse_p)
        plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.2*rangey, 'SEP: %5.3f' % sep)
        plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.25*rangey, 'RPD: %5.3f' % rpd)
        plt.text(min(Y_pred)+0.05*rangex, max(Y_valid)-0.3*rangey, 'Bias: %5.3f' %  bias)
        plt.show()    
prediction(X_calib, Y_calib, X_valid, Y_valid, plot_components=True)
