import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
import timeit

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
    start_time = timeit.default_timer()
    C = r_subset.T@r_subset # correlation matrix, dimension # wavelengths x # wavelengths
    eigvals, eigvecs = sorted_eigs(C, return_eigvalues = True)
    pca_t = timeit.default_timer() - start_time
    print(f"pca time: {pca_t}s")
    for eigi in range(5):
        plt.scatter(logwave, eigvecs[:, eigi], s = 1, label = f"eigenvalue: {eigvals[eigi]}")
    plt.title("First Five Eigenvectors of Correlation Matrix")
    plt.legend()
    plt.savefig("imgs/correlation-5-eigenvec.png")
    plt.clf()
    f = open("res/corr-eigenvals.txt", "w")
    num = len(eigvals)
    for i in range(num):
        f.write(f"{eigvals[i]}\n")
    f.close()
    f = open("res/corr-eigenvectors.csv", "w")
    for i in range(num):
        for j in range(num):
            f.write(f"{eigvecs[j][i]},")
        f.write("\n")
    f.close()

def qe():
    '''
    find eigenvalues by SVD
    '''
    flux_normalized = normalize()
    r = get_normalized_0mean(flux_normalized)
    r_subset = r
    start_time = timeit.default_timer()
    U, S, Vh = np.linalg.svd(r_subset, full_matrices=True)
    # rows of Vh are eigenvectors
    eigvecs_svd = Vh.T # columns are eigenvectors
    eigvals_svd = S**2 
    svd_sort = np.argsort(eigvals_svd)[::-1]
    eigvecs_svd = eigvecs_svd[:,svd_sort]
    eigvals_svd = eigvals_svd[svd_sort]
    svd_t = timeit.default_timer() - start_time
    for eigi in range(5):
        plt.scatter(logwave, eigvecs_svd[:, eigi], s = 1, label = f"eigenvalue: {eigvals_svd[eigi]}")
    plt.title("First Five Eigenvectors by SVD")
    plt.legend()
    plt.savefig("imgs/svd-5-eigenvec.png")
    plt.clf()
    print(f"eigenvectors by svd time: {svd_t}s")
    f = open("res/svd-eigenvals.txt", "w")
    num = len(eigvals_svd)
    for i in range(num):
        f.write(f"{eigvals_svd[i]}\n")
    f.close()
    f = open("res/svd-eigenvectors.csv", "w")
    for i in range(num):
        for j in range(num):
            f.write(f"{eigvecs_svd[j][i]},")
        f.write("\n")
    f.close()

def read_eigen():
    '''
    read eigenvals and eigenvecs from storage
    '''
    eigenvals_corr = np.genfromtxt("res/corr-eigenvals.txt")
    eigenvecs_corr = np.genfromtxt("res/corr-eigenvectors.csv", delimiter = ",")
    eigenvecs_corr = eigenvecs_corr[:4001, :4001]
    eigenvals_svd = np.genfromtxt("res/svd-eigenvals.txt")
    eigenvecs_svd = np.genfromtxt("res/svd-eigenvectors.csv", delimiter = ",")
    eigenvecs_svd = eigenvecs_svd[:4001, :4001]
    return eigenvals_corr, eigenvecs_corr.T, eigenvals_svd, eigenvecs_svd.T

def qe_analysis():
    '''
    compare eigenvectors and eigenvalues by correlation and svd
    '''
    eigenvals_corr, eigenvecs_corr, eigenvals_svd, eigenvecs_svd = read_eigen()
    plt.scatter(eigenvals_corr, eigenvals_svd, s = 5)
    plt.xlabel('Correlation Eigenvalues', fontsize = 16)
    plt.ylabel('SVD Eigenvalues', fontsize = 16)
    plt.title("Eigenvalues Comparison")
    plt.savefig("imgs/eigenvals-compare.png")
    plt.clf()
    for i in range(4001):
        plt.scatter(eigenvecs_corr[i], eigenvecs_svd[i], s = 1)
    plt.xlabel('Correlation Eigevectors', fontsize = 16)
    plt.ylabel('SVD Eigenvectors', fontsize = 16)
    plt.title("Eigenvectors Comparison")
    plt.savefig("imgs/eigenvectors-compare.png")
    plt.clf()


def PCA_dim_reduce(l, r, eigvector, project = True):
    """
    input:
        l: number of eigenvectors selected
        r: residual matrix (normalized, 0 mean)
        eigvector: eigenvectors (of covariance matrix) (sorted by eigenvalues high to low)
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """ 
    eigvec = eigvector[:,:l] # only keep first l eigenvectors
    reduced_wavelength_data = np.dot(eigvec.T,r.T) # coordinates in new basis (l basis vectors which are eigenvectors)
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights (coordinates in new low-dim basis)
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
                                                        # (change coordinates back to previous basis)

def qg():
    '''
    dimensionality reduction
    project data points into the new eigenvector basis (l-dim)
    '''
    flux_normalized = normalize()
    r = get_normalized_0mean(flux_normalized)
    eigenvals_corr, eigenvecs_corr, eigenvals_svd, eigenvecs_svd = read_eigen()
    num_eigen = 5
    approx_data = PCA_dim_reduce(num_eigen, r, eigenvecs_corr)
    # Using only the first eigenvector does not capture the entire signal well! 
    galaxy_ids = [0, 101, 800, 5000]
    for galaxy_i in galaxy_ids:
        plt.plot(logwave, approx_data[galaxy_i,:], label = f"{num_eigen} eigenvectors")
        plt.plot(logwave, r[galaxy_i, :], label = 'original data')
        plt.title(f"Dimensionality Reduction of Galaxy {galaxy_i} (5 Eigenvectors)")
        plt.xlabel('log(wavelength) [$log(A)$]', fontsize = 16)
        plt.ylabel('normalized 0-mean flux', fontsize = 16)
        plt.legend()
        plt.savefig(f"imgs/dim-reduction-5eigenvec-galaxy{galaxy_i}.png")
        plt.clf()

def qh():
    flux_normalized = normalize()
    r = get_normalized_0mean(flux_normalized)
    eigenvals_corr, eigenvecs_corr, eigenvals_svd, eigenvecs_svd = read_eigen()
    num_eigen = 3
    eigvec = eigenvecs_corr[:,:num_eigen] # only keep first l eigenvectors
    new_coordinates = np.dot(eigvec.T,r.T) # coordinates in new basis (l basis vectors which are eigenvectors) l*m
    c0 = new_coordinates[0] # m-dim, projection on first eigenvector for each data point
    c1 = new_coordinates[1]
    c2 = new_coordinates[2]
    plt.scatter(c0, c1, s = 5)
    plt.title("c0-c1")
    plt.xlabel("c0")
    plt.ylabel("c1")
    plt.savefig("imgs/c0-c1.png")
    plt.clf()
    plt.scatter(c0, c2, s = 5)
    plt.title("c0-c2")
    plt.xlabel("c0")
    plt.ylabel("c2")
    plt.savefig("imgs/c0-c2.png")
    plt.clf()

def qi():
    '''
    Compression Error dependent on number of eigenvector selected
    '''
    flux_normalized = normalize()
    r = get_normalized_0mean(flux_normalized)
    eigen_num_list = np.array(list(range(1, 21)))
    squared_residual_list = []
    eigenvals_corr, eigenvecs_corr, eigenvals_svd, eigenvecs_svd = read_eigen()
    eigvec = eigenvecs_corr[:,:20] # only keep first 20 eigenvectors
    new_coordinates = np.dot(eigvec.T,r.T) # coordinates in new basis (l basis vectors which are eigenvectors)
    for eigen_num in eigen_num_list:
        selected_eigenvecs = eigenvecs_corr[:, :eigen_num]
        selected_coords = new_coordinates[:eigen_num, :]
        approx_data = np.dot(selected_eigenvecs, selected_coords).T
        squared_residual_list.append(np.sum(pow(approx_data - r, 2)))
    plt.plot(eigen_num_list, squared_residual_list)
    plt.title("Squared Residuals")
    plt.xlabel("eigenvectors selected")
    plt.ylabel("squared residuals")
    plt.savefig("imgs/squared-residuals.png")
    plt.clf()
    f = open("res/squared-residuals.txt", "w")
    for i in squared_residual_list:
        f.write(f"{i}\n")
    f.close()
    print(f"error when 20 eigenvectors selected: {squared_residual_list[19]}")



# qa()
# qb()
# qc()
# qd()
# qe()
# qe_analysis()
# qg()
# qh()
qi()





    