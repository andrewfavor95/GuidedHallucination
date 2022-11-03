
import os

import pdb as pydebug
import string
import math
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import colabdesign
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants

def parse_ss_spec(ss_spec,outputs,chain_len,copies,helix_cutoff=6.0,sheet_cutoff=9.0):

    dgram = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0,outputs["distogram"]["bin_edges"])

    resi_list = []
    ss_list = []
    bins = jnp.full((dgram.shape[0],dgram.shape[-1]),True)


    for ss_spec_str_i in ss_spec:
        ss_i = ss_spec_str_i.split(',')[0]
        range_i = ss_spec_str_i.split(',')[-1]

        for chain_ind in range(copies):
            start_ind = (chain_ind*chain_len)+int(range_i.split(':')[0])-1
            stop_ind = (chain_ind*chain_len)+int(range_i.split(':')[-1])

            if ss_i=='H':
                bins = bins.at[start_ind:stop_ind,:].set(dgram_bins < helix_cutoff )
            elif ss_i=='S':
                bins = bins.at[start_ind:stop_ind,:].set(dgram_bins > sheet_cutoff )

    return resi_list,ss_list,bins


def rot2quat(matrix, isprecise=False):
	M = jnp.array(matrix, copy=False)[:4, :4]
	if isprecise:
		q = jnp.empty((4, ))
		t = jnp.trace(M)
		if t > M[3, 3]:
			q[0] = t
			q[3] = M[1, 0] - M[0, 1]
			q[2] = M[0, 2] - M[2, 0]
			q[1] = M[2, 1] - M[1, 2]
		else:
			i, j, k = 0, 1, 2
			if M[1, 1] > M[0, 0]:
				i, j, k = 1, 2, 0
			if M[2, 2] > M[i, i]:
				i, j, k = 2, 0, 1
			t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
			q[i] = t
			q[j] = M[i, j] + M[j, i]
			q[k] = M[k, i] + M[i, k]
			q[3] = M[k, j] - M[j, k]
			q = q[[3, 0, 1, 2]]
		q *= 0.5 / math.sqrt(t * M[3, 3])
	else:
		m00 = M[0, 0]
		m01 = M[0, 1]
		m02 = M[0, 2]
		m10 = M[1, 0]
		m11 = M[1, 1]
		m12 = M[1, 2]
		m20 = M[2, 0]
		m21 = M[2, 1]
		m22 = M[2, 2]
		# symmetric matrix K
		K =jnp.array([[m00-m11-m22, 0.0,         0.0,         0.0],
						 [m01+m10,     m11-m00-m22, 0.0,         0.0],
						 [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
						 [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
		K /= 3.0
		# quaternion is eigenvector of K that corresponds to largest eigenvalue
		w, V = jnp.linalg.eigh(K)
		q = V[[3, 0, 1, 2], jnp.argmax(w)]

	flip =  q[0] < 0
	q = jnp.where(flip, -q, q)

	return q




def get_model_num_spec(model_numbers):
	model_num_list = []
	for model_i in model_numbers.split(','):
		model_num_list.append(int(model_i.strip())-1)
		num_models = len(model_num_list)
		
	return model_num_list, num_models





