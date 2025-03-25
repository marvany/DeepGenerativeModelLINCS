# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# # Deep Generative Models on LINCS Data

# Latent representations learning for LINCS L1000 expression profiles with Variational AutoEncoder (VAE) and Supervised Vector-Quantized Variational Autoencoder (S-VQ-VAE).
#
# For a more detailed tutorial of how to S-VQ-VAE on other dataset, please refer to https://github.com/evasnow1992/S-VQ-VAE.

from __future__ import division
import sys
import math
import time
import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmapPy as cp
from cmapPy.pandasGEXpress.parse import parse
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data as Tdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Major cell lines that contain over 10,000 samples from GSE70138 and GSE106127.

cellLineNameSM = set(["A375", "HA1E", "HELA", "HT29", "MCF7", "PC3", "YAPC"])
cellLineNameGP = set(["A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "VCAP"])

# ## Load GSE70138 small molecular perturbation L1000 data

# ### Load signature metadata

sigFileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt"
sigInfoSM = pd.read_csv(sigFileNameSM, sep = "\t")
cellLineInfoSM = sigInfoSM[sigInfoSM["cell_id"].isin(cellLineNameSM)]
cellLineSigSM = cellLineInfoSM["sig_id"]
cellLinePertSM = cellLineInfoSM["pert_id"]
cellLinePertTypeSM = cellLineInfoSM["pert_type"]
print(sigInfoSM.shape)
print(cellLineInfoSM.shape)

pertDicSM = {}
pertArraySM = cellLinePertSM.values
for i in range(len(pertArraySM)):
    pertSM = pertArraySM[i]
    c = pertDicSM.setdefault(pertSM, 0)
    pertDicSM[pertSM] = c + 1
print("The number of perturbagens: " + str(len(pertDicSM)))

# ### Load gene information

geneFileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt"
geneInfoSM = pd.read_csv(geneFileNameSM, sep = "\t")
lmInfoSM = geneInfoSM["pr_gene_id"][geneInfoSM["pr_is_lm"] == 1].astype(str) # landmark genes only
print(geneInfoSM.shape)
print(lmInfoSM.shape)

# ### Load perturbagen class (PCL) information

pclFileNameSM = "../Data/YX_Data/pcl_information.csv"
pertClassDicSM = {}
classDicSM = {}
pertCountSM = 0
with open(pclFileNameSM, "r") as myfile:
    for line in myfile.readlines():
        if not len(line):
            continue
        pertCountSM += 1
        line = line.strip()
        spline = line.split(',')
        pertClassDicSM[spline[0]] = spline[1]
        c = classDicSM.setdefault(spline[1], 0)
        classDicSM[spline[1]] = c + 1
print("The number of perturbagens with class information: " + str(pertCountSM))
print("The number of PCL: " + str(len(classDicSM)))
print(sorted(classDicSM.items(), key = operator.itemgetter(1), reverse = True)[0])

overlapPertDicSM = {}
overLapPertClassDicSM = {}
overLapPertClassCountSM = {}
for pert in pertDicSM:
    if pert in pertClassDicSM:
        overlapPertDicSM[pert] = 1
        c = overLapPertClassDicSM.setdefault(pertClassDicSM[pert], 0)
        overLapPertClassDicSM[pertClassDicSM[pert]] = c + 1
        c = overLapPertClassCountSM.setdefault(pertClassDicSM[pert], 0)
        overLapPertClassCountSM[pertClassDicSM[pert]] = c + pertDicSM[pert]
print("The number of perturbagens with class information: " + str(len(overlapPertDicSM)))
print("The number of classes of overlap perturbagens: " + str(len(overLapPertClassDicSM)))
print(sorted(overLapPertClassDicSM.items(), key = operator.itemgetter(1), reverse = True)[0])
print(sorted(overLapPertClassCountSM.items(), key = operator.itemgetter(1), reverse = True)[0])

# ### Load and process all L1000 data

L1000FileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
gctoo_cellLineSM = parse(L1000FileNameSM, cid = cellLineSigSM, rid = lmInfoSM)
print(gctoo_cellLineSM.data_df.shape)
print(max(gctoo_cellLineSM.data_df.max()))
print(min(gctoo_cellLineSM.data_df.min()))

expDataSM = torch.Tensor(gctoo_cellLineSM.data_df.transpose().values.astype(np.float32))
expDatasetSM = Tdata.TensorDataset(expDataSM)

# ### Load and process L1000 data with PCL information

cellLineInfoSMC = cellLineInfoSM[cellLineInfoSM["pert_id"].isin(pertClassDicSM)]
cellLineSigSMC = cellLineInfoSMC["sig_id"]
gctoo_cellLineSMC = parse(L1000FileNameSM, cid = cellLineSigSMC, rid = lmInfoSM)
print(cellLineInfoSMC.shape)
print(gctoo_cellLineSMC.data_df.shape)

expDataSMC = torch.Tensor(gctoo_cellLineSMC.data_df.transpose().values.astype(np.float32))
pertClassTargetDicSM = {}
pertClassTargetIndexSM = 0
for pert in overlapPertDicSM:
    if pertClassDicSM[pert] not in pertClassTargetDicSM:
        pertClassTargetDicSM[pertClassDicSM[pert]] = pertClassTargetIndexSM
        pertClassTargetIndexSM += 1
pertClassTargetArraySM = np.zeros((expDataSMC.size()[0]))
targetIndex = 0
for sig in gctoo_cellLineSMC.data_df.transpose().index.values:
    pert = cellLineInfoSMC[cellLineInfoSMC["sig_id"] == sig]["pert_id"].values[0]
    pertClass = pertClassDicSM[pert]
    pertClassIndex = pertClassTargetDicSM[pertClass]
    pertClassTargetArraySM[targetIndex] = pertClassIndex
    targetIndex += 1
nClassSMC = len(pertClassTargetDicSM)
print("The number of PCL: " + str(nClassSMC))
pertClassTargetSM = torch.LongTensor(pertClassTargetArraySM)

expDatasetSMC = Tdata.TensorDataset(expDataSMC, pertClassTargetSM)

# ### Load and process L1000 data without perturbagen 'MG-31' and 'bortezomib'

excludePertDicSM = {'MG-132': 1, 'bortezomib': 1}
cellLineInfoSMNP = cellLineInfoSM[~cellLineInfoSM["pert_iname"].isin(excludePertDicSM)]
cellLineSigSMNP = cellLineInfoSMNP["sig_id"]
gctoo_cellLineSMNP = parse(L1000FileNameSM, cid = cellLineSigSMNP, rid = lmInfoSM)
print(cellLineInfoSMNP.shape)
print(gctoo_cellLineSMNP.data_df.shape)

expDataSMNP = torch.Tensor(gctoo_cellLineSMNP.data_df.transpose().values.astype(np.float32))
expDatasetSMNP = Tdata.TensorDataset(expDataSMNP)

# ### Load and process L1000 data with PCL information and without perturbagen 'MG-31'

excludePertDicSMC = {'MG-132': 1}
cellLineInfoSMCNP = cellLineInfoSMC[~cellLineInfoSMC["pert_iname"].isin(excludePertDicSMC)]
cellLineSigSMCNP = cellLineInfoSMCNP["sig_id"]
gctoo_cellLineSMCNP = parse(L1000FileNameSM, cid = cellLineSigSMCNP, rid = lmInfoSM)
print(cellLineInfoSMCNP.shape)
print(gctoo_cellLineSMCNP.data_df.shape)

expDataSMCNP = torch.Tensor(gctoo_cellLineSMCNP.data_df.transpose().values.astype(np.float32))
pertClassTargetArraySMNP = np.zeros((expDataSMCNP.size()[0]))
targetIndex = 0
tempPertDic = {}
for sig in gctoo_cellLineSMCNP.data_df.transpose().index.values:
    pert = cellLineInfoSMCNP[cellLineInfoSMCNP["sig_id"] == sig]["pert_id"].values[0]
    pertClass = pertClassDicSM[pert]
    pertClassIndex = pertClassTargetDicSM[pertClass]
    pertClassTargetArraySMNP[targetIndex] = pertClassIndex
    tempPertDic[pertClassIndex] = 1
    targetIndex += 1
print("The number of PCL: " + str(len(tempPertDic)))
pertClassTargetSMNP = torch.LongTensor(pertClassTargetArraySMNP)
expDatasetSMCNP = Tdata.TensorDataset(expDataSMCNP, pertClassTargetSMNP)

# ## Load GSE106127 genetic perturbation L1000 data

# ### Load signature metadata

sigFileNameGP = "../Data/L1000/GSE106127_sig_info.txt"
sigInfoGP = pd.read_csv(sigFileNameGP, sep = "\t")
cellLineInfoGP = sigInfoGP[sigInfoGP["cell_id"].isin(cellLineNameGP)]
cellLineSigGP = cellLineInfoGP["sig_id"]
cellLinePertGP = cellLineInfoGP["pert_iname"]
cellLinePertTypeGP = cellLineInfoGP["pert_type"]
print(sigInfoGP.shape)
print(cellLineInfoGP.shape)

pertDicGP = {}
pertArrayGP = cellLinePertGP.values
for i in range(len(pertArrayGP)):
    pertGP = pertArrayGP[i]
    c = pertDicGP.setdefault(pertGP, 0)
    pertDicGP[pertGP] = c + 1
print("The number of perturbagens: " + str(len(pertDicGP)))

# ### Load and process L1000 data

L1000FileNameGP = "../Data/L1000/GSE106127_level_5_modz_n119013x978.gctx"
gctoo_cellLineGP = parse(L1000FileNameGP, cid = cellLineSigGP, rid = lmInfoSM)
gctoo_cellLineGP.data_df = gctoo_cellLineGP.data_df.reindex(gctoo_cellLineSM.data_df.index) # reindex to same order of genes
print(gctoo_cellLineGP.data_df.shape)
print(max(gctoo_cellLineGP.data_df.max()))
print(min(gctoo_cellLineGP.data_df.min()))

expDataGP = torch.Tensor(gctoo_cellLineGP.data_df.transpose().values.astype(np.float32))
expDatasetGP = Tdata.TensorDataset(expDataGP)

# ### Merge with SMNP data

dataArraySMNP = expDataSMNP.numpy()
dataArrayGP = expDataGP.numpy()
dataArrayBoth = np.concatenate((dataArraySMNP, dataArrayGP)).astype(np.float32)
expDataBoth = torch.Tensor(dataArrayBoth)
expDatasetBoth = Tdata.TensorDataset(expDataBoth)
print(expDataBoth.shape)

# ### Input data gene order sanity check

print(gctoo_cellLineSM.data_df.index.values[0:8])
print(gctoo_cellLineSMC.data_df.index.values[0:8])
print(gctoo_cellLineSMNP.data_df.index.values[0:8])
print(gctoo_cellLineGP.data_df.index.values[0:8])
print("")
print(gctoo_cellLineSM.data_df.index.values[-8:])
print(gctoo_cellLineSMC.data_df.index.values[-8:])
print(gctoo_cellLineSMNP.data_df.index.values[-8:])
print(gctoo_cellLineGP.data_df.index.values[-8:])
print("")
for i in range(len(gctoo_cellLineSM.data_df.index.values)):
    if gctoo_cellLineSM.data_df.index.values[i] != gctoo_cellLineSMC.data_df.index.values[i] or \
    gctoo_cellLineSM.data_df.index.values[i] != gctoo_cellLineGP.data_df.index.values[i]:
        print(i)
        print(gctoo_cellLineSM.data_df.index.values[i])
        print(gctoo_cellLineSMC.data_df.index.values[i])

# ### PCA check

pca_data = expDataBoth.numpy()

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(pca_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = principalDf

print(finalDf.shape)
print(finalDf.iloc[12797:12799,:])

# +
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(finalDf.loc[:, 'principal component 1']
           , finalDf.loc[:, 'principal component 2']
           , c = 'b'
           , s = 10
           , alpha = 0.9)
ax.grid()
# -

sns.set(style="white", color_codes=True)
snsPlot = sns.JointGrid(x=finalDf["principal component 1"], y=finalDf["principal component 2"])
snsPlot = snsPlot.plot_joint(sns.kdeplot, cmap="Blues_d")
snsPlot = snsPlot.plot_marginals(sns.kdeplot, shade=True)


# ## Variational AutoEncoder

class VAE(nn.Module):
    def __init__(self, hiddenSize = 100):
        super(VAE, self).__init__()
        self.hiddenSize = hiddenSize
        self.encoder = nn.Sequential(
            nn.Linear(978, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(self.hiddenSize, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 978),
            nn.Tanh())
        self.mu = torch.nn.Linear(1000, self.hiddenSize)
        self.logvar = torch.nn.Linear(1000, self.hiddenSize)

    def reparametrize(self, h):
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        h1 = self.encoder[1](self.encoder[0](x))
        h = self.encoder[3](self.encoder[2](h1))
        z, mu, logvar = self.reparametrize(h)
        decoded = self.decoder(z)
        return decoded * 10, mu, logvar, z, h1
    
    def generate_sample(self):
        z = torch.FloatTensor(self.hiddenSize).normal_()
        return self.decoder(z)


reconstruction_function = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar):
    """
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar) # KL(q(z|x)||p(z))
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return BCE + KLD


# ### Start training

num_epochs = 300
learning_rate = 1e-3
batch_size = 512
model_VAE = VAE(hiddenSize = 100)
if torch.cuda.is_available():
    model_VAE.cuda()
optimizer = torch.optim.Adam(model_VAE.parameters(), lr = learning_rate)

useData = 'GP'
if useData == 'SM':
    expTrainDatasetSM, expValidDatasetSM = Tdata.random_split(expDatasetSM, [expDataSM.size()[0] - expDataSM.size()[0]//10, expDataSM.size()[0]//10])
    expTrainLoader = Tdata.DataLoader(dataset = expTrainDatasetSM, batch_size = batch_size, shuffle = True)
    expValidLoader = Tdata.DataLoader(dataset = expValidDatasetSM)
    print(len(expTrainLoader.dataset), len(expValidLoader.dataset))
elif useData == 'GP':
    expTrainDatasetGP, expValidDatasetGP = Tdata.random_split(expDatasetGP, [expDataGP.size()[0] - expDataGP.size()[0]//10, expDataGP.size()[0]//10])
    expTrainLoader = Tdata.DataLoader(dataset = expTrainDatasetGP, batch_size = batch_size, shuffle = True)
    expValidLoader = Tdata.DataLoader(dataset = expValidDatasetGP)
    print(len(expTrainLoader.dataset), len(expValidLoader.dataset))
elif useData == 'both':
    expTrainDatasetBoth, expValidDatasetBoth = Tdata.random_split(expDatasetBoth, [expDataBoth.size()[0] - expDataBoth.size()[0]//10, expDataBoth.size()[0]//10])
    expTrainLoader = Tdata.DataLoader(dataset = expTrainDatasetBoth, batch_size = batch_size, shuffle = True)
    expValidLoader = Tdata.DataLoader(dataset = expValidDatasetBoth)
    print(len(expTrainLoader.dataset), len(expValidLoader.dataset))
else:
    print("Error! Unidentified type of useData!")

epoch_train_loss = []
startTime = time.time()
for epoch in range(num_epochs):
    model_VAE.train()
    lossList = []
    for step, data in enumerate(expTrainLoader):
        optimizer.zero_grad()
        inputData = data[0]
        if torch.cuda.is_available():
            inputData = inputData.cuda()
        recon_batch, mu, logvar, encoded, h1 = model_VAE(inputData)
        loss = loss_function(recon_batch, inputData, mu, logvar)
        loss.backward()
        optimizer.step()
        lossList.append(reconstruction_function(recon_batch, inputData).data.numpy())
    epochLoss = np.mean(lossList)
    epoch_train_loss.append(epochLoss)
    print('Epoch: ', epoch, '| train loss: %.4f' % epochLoss)
timeSpent = round(time.time() - startTime, 2)
print('Done.\nTime spent: ' + str(timeSpent) + 's.')

f = plt.figure(figsize=(8,8))
ax = f.add_subplot(1,1,1)
ax.plot(epoch_train_loss)
ax.set_title('NMSE.')
ax.set_xlabel('epoch')

torch.save(model_VAE.encoder, 'VAE_encode.pth')
torch.save(model_VAE.mu, 'VAE_mu.pth')
torch.save(model_VAE.logvar, 'VAE_logvar.pth')
torch.save(model_VAE.decoder, 'VAE_decode.pth')

# ### Validation on independent dataset

model_VAE.eval()
validLossList = []
for data in expValidLoader:
    inputData = data[0]
    if torch.cuda.is_available():
        inputData = inputData.cuda()
    recon_batch, mu, logvar, _, _ = model_VAE(inputData)
    loss = reconstruction_function(recon_batch, inputData)
    validLossList.append(loss.data.numpy())
print('Validation loss: %.4f' % np.mean(validLossList))

# ## S-VQ-VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### VQ function for S-VQ-VAE

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, divergence_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._divergence_cost = divergence_cost
    
    def forward(self, inputs, label):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.reshape(label,(label.shape[0], 1))
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        close_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        close_encodings = torch.zeros(close_indices.shape[0], self._num_embeddings).to(device)
        close_encodings.scatter_(1, close_indices, 1)
        
        indicator = 1 - (encoding_indices == close_indices)
        indicator = indicator.float()
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight)
        close_quantized = torch.matmul(close_encodings, self._embedding.weight)
        
        # Loss
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        x_latent_loss = torch.mean(indicator * ((close_quantized - inputs.detach())**2))
        d_latent_loss = torch.mean(indicator * ((close_quantized.detach() - inputs)**2))
        loss = q_latent_loss + self._commitment_cost * e_latent_loss - x_latent_loss - self._divergence_cost * d_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings


# ### VQ function for standard VQ-VAE

class VectorQuantizer_normal(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, divergence_cost):
        super(VectorQuantizer_normal, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
    
    def forward(self, inputs, label):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings


class S_VQ_VAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, divergence_cost, decay=0):
        super(S_VQ_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(978, embedding_dim),
            nn.Tanh())
        
        if decay > 0.0:
            self._vq_vae = VectorQuantizer_normal(num_embeddings, embedding_dim, 
                                              commitment_cost, divergence_cost)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost, divergence_cost)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 978),
            nn.Tanh())

    def forward(self, x, label):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self._vq_vae(z, label)
        x_recon = self.decoder(quantized)
        x_recon = x_recon * 10
        return loss, x_recon, perplexity, encodings, quantized


def sparsePenalty(encoded, p):
    q = torch.mean(torch.abs(encoded), dim=0, keepdim=True)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


# ### Start training

# +
embedding_dim = 1000
num_embeddings = nClassSMC
commitment_cost = 0.25
divergence_cost = 0.1
decay = 0

num_epochs = 900
learning_rate = 1e-4
batch_size = 256

model_S_VQ_VAE = S_VQ_VAE(num_embeddings, embedding_dim, commitment_cost, divergence_cost, decay).to(device)
optimizer  = torch.optim.Adam(model_S_VQ_VAE.parameters(), lr=learning_rate, amsgrad=True)
criterion = nn.MSELoss()
# -

expTrainDatasetSMCNP, expValidDatasetSMCNP = Tdata.random_split(expDatasetSMCNP, [expDataSMCNP.size()[0] - expDataSMCNP.size()[0]//10, expDataSMCNP.size()[0]//10])
expTrainLoader = Tdata.DataLoader(dataset = expTrainDatasetSMCNP, batch_size = batch_size, shuffle = True)
expValidLoader = Tdata.DataLoader(dataset = expValidDatasetSMCNP)
print(len(expTrainLoader.dataset), len(expValidLoader.dataset))

expTrainLoader = Tdata.DataLoader(dataset = expDatasetSMCNP, batch_size = batch_size, shuffle = True)
print(len(expTrainLoader.dataset))

train_recon_error = []
train_perplexity = []
startTime = time.time()
for epoch in range(num_epochs):
    model_S_VQ_VAE.train()
    epoch_recon_error = []
    epoch_perplexity = []
    for step, data in enumerate(expTrainLoader):
        optimizer.zero_grad()
        inputData, label = data
        inputData = inputData.to(device)
        label = label.to(device)
        vq_loss, data_recon, perplexity, _, encoded = model_S_VQ_VAE(inputData, label)
        recon_error = criterion(data_recon, inputData)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
    
        epoch_recon_error.append(recon_error.item())
        epoch_perplexity.append(perplexity.item())
    recon_error = np.mean(epoch_recon_error)
    mean_perplexity = np.mean(epoch_perplexity)
    train_recon_error.append(recon_error)
    train_perplexity.append(mean_perplexity)
    print('Epoch %d' % epoch)
    print('recon_error: %.3f' % recon_error)
    print('perplexity: %.3f' % mean_perplexity)
    print('')
timeSpent = round(time.time() - startTime, 2)
print('Done.\nTime spent: ' + str(timeSpent) + 's.')

# +
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_recon_error)
ax.set_title('NMSE.')
ax.set_xlabel('epoch')

ax = f.add_subplot(1,2,2)
ax.plot(train_perplexity)
ax.set_title('Average codebook usage (perplexity).')
ax.set_xlabel('epoch')
# -

torch.save(model_S_VQ_VAE.encoder, 'S_VQ_VAE_encode.pth')
torch.save(model_S_VQ_VAE._vq_vae._embedding, 'S_VQ_VAE_embedding.pth')
torch.save(model_S_VQ_VAE.decoder, 'S_VQ_VAE_decode.pth')

# ### Validation on independent dataset

model_S_VQ_VAE.eval()
validLossList = []
for step, data in enumerate(expValidLoader):
    inputData, label = data
    inputData = inputData.to(device)
    label = label.to(device)
    vq_loss, data_recon, _, _, _= model_S_VQ_VAE(inputData, label)
    recon_error = criterion(data_recon, inputData)
    loss = recon_error + vq_loss
    validLossList.append(recon_error.item())
print('Validation loss: %.4f' % np.mean(validLossList))


