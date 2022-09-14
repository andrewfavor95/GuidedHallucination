
import os
import colabdesign
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
import jax
import jax.numpy as jnp

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




def ss_spec_loss(inputs,outputs,opt):
    
    def ss_probs_for_bins(outputs,
                      alpha_mu=5.1,
                      alpha_sig=0.73,
                      beta_mu=10.2,
                      beta_sig=0.73):

        bin_lowers = jnp.append(0,outputs["distogram"]["bin_edges"])
        bin_uppers = jnp.append(outputs["distogram"]["bin_edges"],1e3)

        P_ss_by_bin = {'helix':jax.scipy.stats.norm.cdf(bin_uppers, alpha_mu, alpha_sig) - jax.scipy.stats.norm.cdf(bin_lowers, alpha_mu, alpha_sig),
                       'strand':jax.scipy.stats.norm.cdf(bin_uppers, beta_mu, beta_sig) - jax.scipy.stats.norm.cdf(bin_lowers, beta_mu, beta_sig)}

        return P_ss_by_bin
  


    def parse_ss_spec(ss_spec,outputs,chain_len,copies,helix_cutoff=6.0,sheet_cutoff=9.0):


        P_ss_by_bin = ss_probs_for_bins(outputs)
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
    
    
    chain_len = af_model._len
    copies= af_model._args["copies"]

    dgram = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0,outputs["distogram"]["bin_edges"])

    bin_ss_probs = parse_ss_spec(ss_spec,outputs,chain_len,copies)

    dgram_diag = jnp.diagonal(dgram,offset=3,axis1=0,axis2=1).T

    dgram_smooth = jnp.zeros((dgram.shape[0],dgram.shape[-1]))
    
    for i in range(4):
        dgram_smooth = dgram_smooth.at[i:chain_len+i-3,:].set((dgram_smooth[i:chain_len+i-3,:]+dgram_diag))

    px = jax.nn.softmax(dgram_smooth) # probability of a given residue (axis 0) being in a given bin (axis 1)
    
    correct_ss_prob = (px*bin_ss_probs).sum(-1) # gives array where sum of axis -1 elements is probability of being in the CORRECT secondary structure
    correct_ss_ent = -jnp.log(correct_ss_prob + 1e-8)

    ss_spec_loss_val = correct_ss_ent.mean()

    return {"ss_spec_loss":ss_spec_loss_val}


