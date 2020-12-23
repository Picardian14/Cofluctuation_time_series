import scipy.io
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ts = scipy.io.loadmat('ts.mat')['ts']

#Tomamos la longitud de la medicion, ntime
#y la cantidad de nodos o voxels
#calculamos con eso la cantidad de aristas


zscore_ts = stats.zscore(ts)

ntime, nnodes = ts.shape

nedges = nnodes*(nnodes -1) / 2


#Aca calculan los indices como arreglo
#de una matriz triangular de tamanio nnodes
idx = np.triu_indices(nnodes, k=1)

fc = np.corrcoef(zscore_ts, rowvar=False) #Las columnas son los voxels

ets = zscore_ts[:,idx[0]]*zscore_ts[:,idx[1]]
mu_ets = np.mean(ets, axis=0)
#Si quisiera imitar como orden matlab:
#fc_upper_triangle = fc.T[np.tril(nnodes, k=-1).astype('bool')]

fc_upper_triangle = fc[idx]
plt.plot(fc_upper_triangle, mu_ets)
plt.xlabel('Upper Triangle fc matrix')
plt.ylabel('time average of edge time series')
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.savefig('Time_avereged_edgetsxuppper_triangle.png')

#aplitud de cofluctuacion
rms = np.sqrt(np.sum(ets**2, axis=1))


#frackeep es el porcentaje de frames de alta/baja amplitud a mantener
frackeep = 0.1
nkeep = round(ntime*frackeep)

#sorted_rms = rms[::-1].sort()
idxsort = np.argsort(rms)[::-1]

fctop = np.corrcoef(zscore_ts[idxsort[:nkeep],:], rowvar=False)
fcbot = np.corrcoef(zscore_ts[idxsort[-nkeep:],:], rowvar=False)

#PLOTEO
#fig, axes = plt.subplots(nrows=3, ncols=2)
#axes[][1].plot(rms)
#axes[1].set_xlabel('frames')
#axes[1].set_ylabel('rss')
#axes[0].imshow(fc, vmin=-1, vmax=1)
#axes[0].set_title('fc all time points')
#axes[2].imshow(fctop, vmin=-1, vmax=1)
#axes[2].set_title('fc high amplitude time points')
#sns.scatterplot()
#
#sns.scatter(fctop[idx],fc[idx],ax=axes[3],  palette=['black'])
r_high = np.corrcoef(fctop[idx], fc[idx])[0,1]
# similiraidad entre correlacion en todos los frames y solo con los high/low

rho = [np.corrcoef(fctop[idx], fc[idx]), np.corrcoef(fcbot[idx], fc[idx])]
#mask = np.triu(np.ones_like(corr, dtype=bool))

mask = np.triu(np.ones_like(fctop, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(fcbot, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True)

f.savefig('fc_top_heatmap.png')