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

# # Drug gene target prediction handle

from __future__ import division
import operator
import collections
import numpy as np
import pandas as pd
import cmapPy as cp
from cmapPy.pandasGEXpress.parse import parse
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data as Tdata
from sklearn.neighbors import NearestNeighbors

# ## 1. Load and prepare data

# Major cell lines that contain over 10,000 samples from GSE70138 and GSE106127.

cellLineNameSM = set(["A375", "HA1E", "HELA", "HT29", "MCF7", "PC3", "YAPC"])
cellLineNameGP = set(["A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "VCAP"])

# ### 1.1. Load GSE70138 small molecular L1000 data

# #### 1.1.1. Load signature metadata

sigFileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt"
sigInfoSM = pd.read_csv(sigFileNameSM, sep = "\t")
cellLineInfoSM = sigInfoSM[sigInfoSM["cell_id"].isin(cellLineNameSM)]
cellLineSigSM = cellLineInfoSM["sig_id"]
cellLinePertSM = cellLineInfoSM["pert_iname"]
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

# #### 1.1.2. Load gene information

geneFileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt"
geneInfoSM = pd.read_csv(geneFileNameSM, sep = "\t")
lmInfoSM = geneInfoSM["pr_gene_id"][geneInfoSM["pr_is_lm"] == 1].astype(str) # landmark genes only
print(geneInfoSM.shape)
print(lmInfoSM.shape)

# #### 1.1.3. Load and process all L1000 data

L1000FileNameSM = "../Data/L1000/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
gctoo_cellLineSM = parse(L1000FileNameSM, cid = cellLineSigSM, rid = lmInfoSM)
print(gctoo_cellLineSM.data_df.shape)
print(max(gctoo_cellLineSM.data_df.max()))
print(min(gctoo_cellLineSM.data_df.min()))

expDataSM = torch.Tensor(gctoo_cellLineSM.data_df.transpose().values.astype(np.float32))
expDatasetSM = Tdata.TensorDataset(expDataSM)

# ### 1.2. Load GSE106127 genetic perturbagen L1000 data

# #### 1.2.1. Load signature metadata

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

# #### 1.2.2. Load and process L1000 data

L1000FileNameGP = "../Data/L1000/GSE106127_level_5_modz_n119013x978.gctx"
gctoo_cellLineGP = parse(L1000FileNameGP, cid = cellLineSigGP, rid = lmInfoSM)
gctoo_cellLineGP.data_df = gctoo_cellLineGP.data_df.reindex(gctoo_cellLineSM.data_df.index) # reindex to same order of genes
print(gctoo_cellLineGP.data_df.shape)
print(max(gctoo_cellLineGP.data_df.max()))
print(min(gctoo_cellLineGP.data_df.min()))

expDataGP = torch.Tensor(gctoo_cellLineGP.data_df.transpose().values.astype(np.float32))
expDatasetGP = Tdata.TensorDataset(expDataGP)

# ## 2. Drug gene target prediction

# ### 2.1. Load the SMGP trained VAE model

VAE_encoder = torch.load("NN_model_multiCellLine/VAE_encode_both_train_All.pth")
VAE_decoder = torch.load("NN_model_multiCellLine/VAE_decode_both_train_All.pth")
VAE_mu = torch.load("NN_model_multiCellLine/VAE_mu_both_train_All.pth")
sigNodeArray = np.array([4, 21, 35, 50, 51, 54, 56, 76, 81, 85, 87, 97])

# ### 2.2. Drugs of interest and their ground truth targets

chemGeneDic = {
    'bortezomib': set(['PSMA1','PSMA3','PSMA5','PSMA8','PSMB1','PSMB10','PSMB5','PSMB7']),
    'pitavastatin': set(['HMGCR']),
}
useLayer = 'e3' # choose from 'r', 'e1', 'e2', 'e3', 'sig', 'd1', 'd2'

geneSet = set()
for drug in chemGeneDic:
    for gene in chemGeneDic[drug]:
        geneSet.add(gene)

# ### 2.3. Check availability of drugs and genes in the dataset

validDrugSet = set()
validGeneSet = set()
for drug in chemGeneDic:
    if drug in pertDicSM:
        validDrugSet.add(drug)
for gene in geneSet:
    if gene in pertDicGP:
        validGeneSet.add(gene)
print("Drugs that are available in the dataset:", validDrugSet)
print("Genes that are available in the dataset:", validGeneSet)

validChemGeneDic = {}
for drug in chemGeneDic:
    if drug not in validDrugSet:
        continue
    validGenes = [gene for gene in chemGeneDic[drug] if gene in validGeneSet]
    if len(validGenes) == 0:
        continue
    validChemGeneDic[drug] = set(validGenes)
print("The drugs to be searched and their known targets:")
print(validChemGeneDic)

validGeneChemDic = collections.defaultdict(set)
for drug in validChemGeneDic:
    for gene in validChemGeneDic[drug]:
        validGeneChemDic[gene].add(drug)

# ### 2.4. Drug-target prediction with a representation type

nNei = expDataGP.shape[0]
print("The total number of gene knock down samples to be compared with:", nNei)
print("")
for drug in validChemGeneDic:
    infoDrug = cellLineInfoSM[cellLineInfoSM["pert_iname"] == drug]
    indexList = []
    drugSig = infoDrug["sig_id"]
    drugData = parse(L1000FileNameSM, cid = drugSig, rid = lmInfoSM)
    drugData = drugData.data_df.transpose().values.astype(np.float32)
    
    if useLayer == "e1": # Encoder layer 1
        h = VAE_encoder[1](VAE_encoder[0](torch.Tensor(drugData)))
    elif useLayer == "e2": # Encoder layer 2
        h = VAE_encoder(torch.Tensor(drugData))
    elif useLayer == "e3": # Top hidden layer
        h = VAE_encoder(torch.Tensor(drugData))
        h = VAE_mu(h)
    elif useLayer == "sig": # Signature nodes
        h = VAE_encoder(torch.Tensor(drugData))
        h = VAE_mu(h)
        h = h [:,sigNodeArray]
    elif useLayer == "d1": # Decoder layer 1
        h = VAE_encoder(torch.Tensor(drugData))
        h = VAE_mu(h)
        h = VAE_decoder[1](VAE_decoder[0](h))
    elif useLayer == "d2": # Decoder layer 2
        h = VAE_encoder(torch.Tensor(drugData))
        h = VAE_mu(h)
        h = VAE_decoder[1](VAE_decoder[0](h))
        h = VAE_decoder[3](VAE_decoder[2](h))
    if useLayer != "r":
        drugData = h.data.numpy()

    geneData = expDataGP.data.numpy()
    if useLayer == "e1":
        h = VAE_encoder[1](VAE_encoder[0](torch.Tensor(geneData)))
    elif useLayer == "e2":
        h = VAE_encoder(torch.Tensor(geneData))
    elif useLayer == "e3":
        h = VAE_encoder(torch.Tensor(geneData))
        h = VAE_mu(h)
    elif useLayer == "sig":
        h = VAE_encoder(torch.Tensor(geneData))
        h = VAE_mu(h)
        h = h [:,sigNodeArray]
    elif useLayer == "d1":
        h = VAE_encoder(torch.Tensor(geneData))
        h = VAE_mu(h)
        h = VAE_decoder[1](VAE_decoder[0](h))
    elif useLayer == "d2":
        h = VAE_encoder(torch.Tensor(geneData))
        h = VAE_mu(h)
        h = VAE_decoder[1](VAE_decoder[0](h))
        h = VAE_decoder[3](VAE_decoder[2](h))
    if useLayer != "r":
        geneData = h.data.numpy()

    nbrs = NearestNeighbors(n_neighbors=nNei, algorithm='brute', metric = 'correlation').fit(geneData)
    distances, indices = nbrs.kneighbors(drugData)
    for i in range(indices.shape[0]):
        for j in range(nNei):
            ind = indices[i][j]
            if cellLinePertGP[ind] in validChemGeneDic[drug]:
                indexList.append(j)
                break
    print("Drug:", drug)
    print("The number of samples this drug tested on:", len(indexList))
    print("The top10 ranks of top ranked target genes:", sorted(indexList)[:10])
    print("The mean ranks of top ranked target genes:", round(np.mean(indexList), 3))
    print("")


