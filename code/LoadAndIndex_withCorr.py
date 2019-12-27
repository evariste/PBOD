#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:10:06 2019

@author: Jonathan
"""

##Basic script to load in a mask, find it's nearest neighbours 
#This takes about 20 seconds for a 10mil vox image, 1.5mil mask, about 3.5G max to 2.0G all int, 5s for baby scale

import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import shift
from sklearn.metrics.pairwise import euclidean_distances


#Load the mask image and index coords
img=nib.load('/Users/Jonathan/Projects/GProcess/Paper/mask_restrict.nii')
mask=img.get_data()
mask=np.ndarray.astype(mask,'int16')
mask_index=np.ndarray.flatten(mask)
n_voxels=np.sum(mask_index)
mask_coords=np.argwhere(mask)


#Find neighbours (width set by n_nbh) Original Mah paper just used 1 vox neighbourhood
n_nbh=1
nbhd=(1+(2*n_nbh))


#Does the neighbour exist in the mask
vector_size=np.shape(mask_index)[0]
out_index=np.zeros((vector_size,nbhd,nbhd,nbhd),dtype=int)
out_index_coords=np.zeros((vector_size,nbhd,nbhd,nbhd,3),dtype=int)

#Get all neighbourhood voxels - uses shift - is fast - but is not memory efficient 
for x in np.arange(0-n_nbh,n_nbh+1):
    for y in np.arange(0-n_nbh,n_nbh+1):
        for z in np.arange(0-n_nbh,n_nbh+1):
            #Shift mask in all 3 axes and check the intersection with the search mask exists
            out_index[:,x+n_nbh,y+n_nbh,z+n_nbh]=(np.ndarray.flatten((shift(mask, [x, y, z], output=None, order=0, mode='constant', cval=0.0, prefilter=False)+mask))>1)
            out_index_coords[(mask_index>0),x+n_nbh,y+n_nbh,z+n_nbh,:]=np.hstack(((mask_coords[:,0:1]+x),(mask_coords[:,1:2]+y),(mask_coords[:,2:3]+z)))

out_index=np.reshape(out_index[mask_index>0,:,:],(n_voxels,nbhd**3))
out_index_coords=np.reshape(out_index_coords[mask_index>0,:,:,:],(n_voxels,nbhd**3,3))
out_index_coords=np.flip(out_index_coords,1) #Indexed in the wrong direction in reshape... 


#Make everything into a list for funtime indexing similarity measures
#Each array in this list has a list of neighbouring voxel coords for each input voxel
fun=[]
for i in np.arange(0,n_voxels):
    fun.append(out_index_coords[i,out_index[i,:]>0,:])
    

    

##Step 2 
##Calculate distance measures on the actual data
#To get the values in mask for voxel 12
img=nib.load('/Users/Jonathan/Projects/GProcess/Paper/ModelNoScale/orig_T2w_time2_z.nii.gz');
raw_data=img.get_data()

#Get values for euclidean / correlation - need to figure out calculating distance to knn aspect & zeta 
#Mahalanobis is really slow for me but I think (know) I'm doing it wrong
for j in np.arange(0,n_voxels):
    test=np.corrcoef(raw_data[fun[j][:,0],fun[j][:,1],fun[j][:,2],:].T)
    test=euclidean_distances(raw_data[fun[42][:,0],fun[42][:,1],fun[42][:,2],:].T)
    print(j/n_voxels)



