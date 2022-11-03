
import os

import pdb as pydebug
import string
import math

import numpy as np
# import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import colabdesign
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants





def xform_diff_loss(inputs,outputs,opt,center_coords=True):

	copies = len(jnp.unique(inputs["sym_id"]))
	chain_len = int(len(inputs["sym_id"])/copies)

	pred = outputs["structure_module"]["final_atom_positions"][:,residue_constants.atom_order["CA"]]
	
	centroid = pred.mean(axis=0)
  
	sub_pred_dict = {}
	
	for i in range(copies):
		start_i = chain_len*i
		stop_i = chain_len*(i+1)
		
		if center_coords:
			sub_pred_dict[i] = pred[start_i:stop_i,:]-centroid
		else:
			sub_pred_dict[i] = pred[start_i:stop_i,:]
			
	chain_order_sorted = [i for i in range(copies)]

	# Define different chain pairs to compare transforms between
	xform_pairs = [(chain_order_sorted[i%copies],chain_order_sorted[j%copies]) for i,j in zip(range(0,copies),range(1,copies+1))]

	# Start gathering the info for all the chain transformations
	rmsd_list = []
	quat_list = []

	# for chain_i,chain_j in xform_pairs:
	
	for i,j in xform_pairs:
		R_ij = colabdesign.shared.protein._np_kabsch(sub_pred_dict[i],sub_pred_dict[j])
		rmsd_ij = colabdesign.shared.protein._np_rmsd(sub_pred_dict[i],sub_pred_dict[j])
		q_ij = rot2quat(R_ij)
		
		quat_list.append(q_ij)
		rmsd_list.append(rmsd_ij)



	chain_rmsd_loss_val = jnp.square(jnp.array(rmsd_list)).sum()/len(rmsd_list)

	xform_diff_loss_val = jnp.square(jnp.std(jnp.array(quat_list),axis=0)).sum()
	
	
	return {"chain_rmsd_loss": chain_rmsd_loss_val, "xform_diff_loss": xform_diff_loss_val,  }



def aspect_ratio_loss(inputs,outputs,opt):


	pred = outputs["structure_module"]["final_atom_positions"][:,residue_constants.atom_order["CA"]]
	
	
	s = jnp.linalg.svd(pred - pred.mean(axis=0),full_matrices=True,compute_uv=False)  # singular values of the coordinates

	aspect_ratio_loss_val = jnp.std(s[:3])/jnp.mean(s[:3])
	
	return {"aspect_ratio_loss": aspect_ratio_loss_val }





def domain_contact_loss(inputs,outputs,opt, 
	w=0.075, # well-steepness of attractive potential
	r=4.8, # equilibrium distance of backbone coordinates
	n=1.5, # changes curvature order of potential well
	seq_cutoff=2, # sequence distance cutoff for making mask over contact matrix
	chain_self_weight = -0.3,# relative weight of contacts within same chain (can be negative if want repulsion in same chain, but should downweight self-seq closeness then)
	domain_frac_reweight = True):



	def dist_potential(x,w=0.075,r=4.8,n=1.5):
		return ( (w*(x-r))*jnp.log(x/r)  )**n

	copies = len(jnp.unique(inputs["sym_id"]))
	chain_len = int(len(inputs["sym_id"])/copies)

	
	xyz_array = outputs["structure_module"]["final_atom_positions"][:,residue_constants.atom_order["CA"]]
	# copies=af_model._copies
	# chain_len=af_model._len

	
	model_dmat = jnp.square(xyz_array[:,None]-xyz_array[None,:]).sum(-1)
	
	pair_weight_mat = jnp.ones(model_dmat.shape)
	
	icc_pot_mat = jnp.ones(model_dmat.shape)
	domain_contact_loss = 0.0
	
	for i in range(copies):
		for j in range(copies):
			if i==j:
				pair_weight_mat = pair_weight_mat.at[i*chain_len:(i+1)*chain_len,j*chain_len:(j+1)*chain_len].set(chain_self_weight)
			else:
				pair_weight_mat = pair_weight_mat.at[i*chain_len:(i+1)*chain_len,j*chain_len:(j+1)*chain_len].set(1.0 - jnp.abs(chain_self_weight))
		
	
	
	seq_dist_mask = 1.0*(jnp.square(jnp.arange(copies*chain_len).T[:,None]-jnp.arange(copies*chain_len).T[None,:]) > seq_cutoff**2 )
	
	contact_pot_mat = dist_potential(model_dmat+1e-8,w,r,n)
	
	all_score_mat = pair_weight_mat*contact_pot_mat*seq_dist_mask
	
	domain_contact_loss_val = jnp.sum(all_score_mat)/((chain_len**4)*(copies**2))

#     return domain_contact_loss
	return {"domain_contact_loss": domain_contact_loss_val}





# def ss_spec_loss(inputs,outputs,
def ss_spec_loss(inputs,outputs,ss_spec,
                 resi_buffer=2, # number of buffer residues to ignore when detecting secondary structure errors
                 helix_cutoff=6.0, # upper limit of distances between CA atoms of residue i and residue i+3 for alpha helices
                 sheet_cutoff=9.5  # lower limit of distances between CA atoms of residue i and residue i+3 for beta strands
                 ):
      

    def parse_ss_spec(ss_spec,outputs,chain_len,copies,helix_cutoff=6.0,sheet_cutoff=9.5):

        bin_lowers = jnp.append(0,outputs["distogram"]["bin_edges"])
        bin_uppers = jnp.append(outputs["distogram"]["bin_edges"],1e3)

        P_ss_by_bin = {'helix':jnp.logical_or((bin_lowers < helix_cutoff),(bin_uppers < helix_cutoff)),
                      'strand':jnp.logical_or((bin_lowers > sheet_cutoff),(bin_uppers > sheet_cutoff))}

        dgram = outputs["distogram"]["logits"]
        bin_ss_probs = jnp.ones((dgram.shape[0],dgram.shape[-1]))

        for ss_spec_str_i in ss_spec:
            ss_i = ss_spec_str_i.split(',')[0]
            range_i = ss_spec_str_i.split(',')[-1]

            for chain_ind in range(copies):
                start_ind = (chain_ind*chain_len)+int(range_i.split(':')[0])-1
                stop_ind = (chain_ind*chain_len)+int(range_i.split(':')[-1])

                if ss_i=='H':
                    bin_ss_probs = bin_ss_probs.at[start_ind:stop_ind,:].set(P_ss_by_bin["helix"])

                elif ss_i=='S':
                    bin_ss_probs = bin_ss_probs.at[start_ind:stop_ind,:].set(P_ss_by_bin["strand"])

        return bin_ss_probs



    # chain_len = af_model._len
    # copies= af_model._args["copies"]
    # chain_len = inputs["residue_index"].shape[0]
    copies = len(jnp.unique(inputs["sym_id"]))
    chain_len = int(len(inputs["sym_id"])/copies)

    target_resis = jnp.zeros(chain_len)

    for sse in ss_spec:
      sse_type, sse_range = sse.split(',')
      start = int(sse_range.split(':')[0])-1

      stop = int(sse_range.split(':')[-1])

      if not sse_type == 'L':
        if stop-2>start+2:
          target_resis = target_resis.at[start+2:stop-2].set(1)
        else:
          target_resis = target_resis.at[start:stop].set(1)

    dgram = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0,outputs["distogram"]["bin_edges"])

    bin_ss_probs = parse_ss_spec(ss_spec,outputs,chain_len,copies)


    resdex_offset = 3

    dgram_diag = jnp.diagonal(dgram,offset=resdex_offset,axis1=0,axis2=1).T

    dgram_smooth = jnp.zeros((dgram.shape[0],dgram.shape[-1]))
    
    for i in range(resdex_offset+1):
        dgram_smooth = dgram_smooth.at[i:chain_len+i-resdex_offset,:].set( dgram_smooth[i:chain_len+i-resdex_offset,:] + (0.25*dgram_diag) )

    px = jax.nn.softmax(dgram_smooth) 
    
    correct_ss_prob = (px * bin_ss_probs).sum(-1) 
    correct_ss_ent = -jnp.log(correct_ss_prob + 1e-8)


    ss_spec_loss_val = (correct_ss_ent * target_resis).sum()/(chain_len)  


    return {"ss_spec_loss":ss_spec_loss_val}

    


# def contact_spec_loss(inputs,outputs):
def contact_spec_loss(inputs,outputs,contact_spec):

    def parse_contact_spec(contact_spec,outputs,chain_len,copies):

        contact_range_list = []
        for contact_str_i in contact_spec:
            row_range_i = contact_str_i.split(',')[0]
            col_range_i = contact_str_i.split(',')[-1]

            for chain_ind in range(copies):
                row_start_ind = (chain_ind * chain_len) + int(row_range_i.split(':')[0])-1
                row_stop_ind = (chain_ind * chain_len) + int(row_range_i.split(':')[-1])

                col_start_ind = (chain_ind * chain_len) + int(col_range_i.split(':')[0])-1
                col_stop_ind = (chain_ind * chain_len) + int(col_range_i.split(':')[-1])

                contact_range_list.append(((row_start_ind,row_stop_ind ), (col_start_ind,col_stop_ind )))

        return contact_range_list


    def min_k(x, k=1, mask=None):
        y = jnp.sort(x if mask is None else jnp.where(mask,x,jnp.nan))
        k_mask = jnp.logical_and(jnp.arange(y.shape[-1]) < k, jnp.isnan(y) == False)
        return jnp.where(k_mask,y,0).sum(-1) / (k_mask.sum(-1) + 1e-8)


    def dist_potential(x,w=0.1,r=7.5,n=2.5):
        return ( w*(x-r)*jnp.log(x/r)  )**n

    # chain_len = af_model._len
    # copies= af_model._args["copies"]
    copies = len(jnp.unique(inputs["sym_id"]))
    chain_len = int(len(inputs["sym_id"])/copies)

    dist_logits = outputs["distogram"]["logits"]
    dist_bins = jnp.append(0,outputs["distogram"]["bin_edges"])

    px = jax.nn.softmax(dist_logits)
    dm = jnp.inner(px,dist_bins)



    contact_range_list = parse_contact_spec(contact_spec,outputs,chain_len,copies)

    contact_spec_loss_val = jnp.square(0)

    for (from_i,to_i),(from_j,to_j) in contact_range_list:

        contact_spec_loss_val += min_k(
                                    min_k(
                                        dist_potential(
                                            dm[from_i:to_i,from_j:to_j]+1e-8),
                                        k=2), 
                                    k=(to_j-from_j)
                                    )


        contact_spec_loss_val += min_k(
                                    min_k(
                                        dist_potential(
                                            dm[from_j:to_j,from_i:to_i]+1e-8),
                                        k=2), 
                                    k=(to_i-from_i)
                                    )
    
    contact_spec_loss_val = contact_spec_loss_val/len(contact_range_list)


    return {"contact_spec_loss":contact_spec_loss_val }



