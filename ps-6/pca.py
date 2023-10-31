import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np

hdu_list = astropy.io.fits.open("specgrid.fits")
logwave = hdu_list["LOGWAVE"].data
flux = hdu_list["FLUX"].data

def plot_galaxy(galaxy_i):
    plt.plot(logwave, flux[galaxy_i, :])
    plt.xlabel('log(wavelength) [$log(A)$]', fontsize = 16)
    plt.ylabel('flux [$10^{−17}$ erg s$^{−1}$ cm$^{−2}$ A$^{-1}$]', fontsize = 16)
    plt.title(f"Galaxy {galaxy_i}")
    plt.savefig(f"imgs/galaxy{galaxy_i}.png")
    plt.clf()

def qa():
    print(f"shape of logwave array: {logwave.shape}")             
    print(f"shape of flux array: {flux.shape}")
    plot_galaxy(80)
    plot_galaxy(800)
    plot_galaxy(8000)

def normalize():
    # find normalization over wavelength for each galaxy
    flux_sum = np.sum(flux, axis = 1)
    flux_normalized = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T
    return flux_normalized

def qb():
    flux_normalized = normalize()
    # check that the data is properly "normalized"
    plt.plot(np.sum(flux_normalized, axis = 1))
    plt.ylim(0,2)
    plt.title("Nomalization Check")
    plt.savefig("imgs/normalization-check.png")
    plt.clf()

def get_normalized_0mean(flux_normalized):
    # subtract off mean
    means_normalized = np.mean(flux_normalized, axis=1)
    flux_normalized_0mean = flux_normalized-np.tile(means_normalized, (np.shape(flux)[1], 1)).T
    return flux_normalized_0mean

def qc():
    flux_normalized = normalize()
    flux_normalized_0mean = get_normalized_0mean(flux_normalized)
    galaxy_i = 100
    fig = plt.figure(figsize=(5.5,3),dpi=300)
    plt.plot(logwave, flux_normalized_0mean[galaxy_i,:])
    plt.xlabel('log(wavelength) [$log(A)$]', fontsize = 10)
    plt.ylabel('normalized 0-mean flux', fontsize = 10)
    plt.title(f"Normalized 0-Mean Galaxy {galaxy_i}",fontsize = 10)
    fig.tight_layout()
    plt.savefig(f"imgs/normalized-0mean-galaxy{galaxy_i}.png")
    plt.clf()
    
def sorted_eigs(mat, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues (reverse order of eigenvalue) 
    of matrix mat
    return sorted (high to low) eigenvalues and eigenvectors (each column is an eigenvector)
    -----------------------------------------------------
    """
    eigs=np.linalg.eig(mat) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec

def qd():
    '''
    Principle Component Analysis of correlation matrix
    '''
    flux_normalized = normalize()
    r = get_normalized_0mean(flux_normalized)
    # r_subset = r[:500, :] # a subset for test
    r_subset = r
    # logwave_subset = logwave
    C = r_subset.T@r_subset # correlation matrix, dimension # wavelengths x # wavelengths
    eigvals, eigvecs = sorted_eigs(C, return_eigvalues = True)
    for eigi in range(5):
        plt.scatter(logwave, eigvecs[:, eigi], s = 1, label = f"eigenvalue: {eigvals[eigi]}")
    plt.title("First Five Eigenvectors of Correlation Matrix")
    plt.legend()
    plt.savefig("imgs/correlation-5-eigenvec.png")
    plt.clf()


# qa()
# qb()
# qc()
# qd()




    