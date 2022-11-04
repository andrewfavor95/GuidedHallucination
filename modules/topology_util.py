
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


letter_to_row = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V':21, 'W':22, 'X':23, 'Y':24, 'Z':25 }


ss_to_rgb = {'H':(0.00, 0.86, 0.83, 0.70),
             'S':(0.85, 0.30, 0.85, 0.70)}

def get_topo_params(topo_string,k=3,plot_diagram=False):

  
  def plot_cartoon(row_list,col_list,marker_list,ss_color_list):
    # define weaving direction vectors
    u_list = []
    v_list = []
    for x_i,y_i,x_j,y_j in zip(col_list[:-1],row_list[:-1],col_list[1:],row_list[1:]):
      u_list.append(x_j-x_i)
      v_list.append(y_j-y_i)
    # Plot shit
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.quiver(col_list[:-1],row_list[:-1],u_list,v_list, scale=1.05, units='xy')
    for x,y,c,o in zip(col_list, row_list, ss_color_list, marker_list):
      ax1.scatter(x, y, s=500, color=c, marker="o",cmap='cool')
      ax1.scatter(x, y, s=200, c="k", marker=o)
    ax1.set_aspect('equal')
    plt.show()


  row_list = []
  col_list = []
  ss_list = []
  ori_list = []

  ss_color_list = []
  marker_list = []

  sse_per_row = {row_letter:0 for row_letter in list(letter_to_row.keys())}

  lattice = np.zeros((3,10))

  sse_list = topo_string.split('.')

  

  for sse in sse_list:


    # Get the row/layer
    row = letter_to_row[sse[1]]

    # Get col/relative position in layer
    col = int(sse[2:])

    # Get the SS type
    if row % 2 == 0:
      ss_type = 'H'
    else:
      ss_type = 'S'

    # Get orientation of sse
    ori = sse[0]
    
    # Update lists:
    row_list.append(row)
    col_list.append(col)
    ss_list.append(ss_type)
    ori_list.append(ori)

    sse_per_row[sse[1]] += 1

    # Stuff for plot
    if ori=='+':
      marker_list.append('+')
    elif ori=='-':
      marker_list.append('_')


    ss_color_list.append(ss_to_rgb[ss_type])
  
  
  # Find most populated ss row
  max_row_size = max(sse_per_row.values())
  max_row_letter = max(sse_per_row, key=sse_per_row.get)
  max_row = letter_to_row[max_row_letter]
  n = len(col_list)

  if plot_diagram:
    plot_cartoon(row_list,col_list,marker_list,ss_color_list)

  # Update column positions to be evenly spread
  col_list_old = col_list.copy()
  for i in reversed(range(1,len(row_list)-1)):
    row_i = row_list[i]
    col_i = col_list[i]

    next_col = col_list[i+1]
    prev_col = col_list[i-1]

    if not row_i==max_row:
      col_list[i] = (next_col + prev_col)/2.0

  if plot_diagram:
    plot_cartoon(row_list,col_list,marker_list,ss_color_list)


  xy_coords = jnp.array([col_list,row_list]).T
  dmat = jnp.square(xy_coords[:,None]-xy_coords[None,:]).sum(-1)
  neighbor_dists = dmat + 1e4*jnp.eye(n)

  knn_inds = []
  sorted_inds = jnp.argsort(neighbor_dists)
  for i,neighbors_i in enumerate(sorted_inds):
    if neighbor_dists[i,neighbors_i[k-1]]==neighbor_dists[i,neighbors_i[k]]:
      knn_inds.append(neighbors_i[:k+1].tolist())
    else:
      knn_inds.append(neighbors_i[:k].tolist())


  return ss_list, ori_list, knn_inds





def get_design_params(topo_string, N_tot=None,sse_scale=1.0,k=3,plot_diagram=False):

  ss_list, ori_list, knn_inds = get_topo_params(topo_string,k=k,plot_diagram=plot_diagram)


  X_ab = 2.333 # ratio of alpha angstroms/AA to beta angstroms/AA
  a_l = 3 # initial number of aa per loop
  n_a = 0
  n_b = 0
  looped_string = ''

  for _ in ss_list:
    if _ =='H':
      n_a+=1
    elif _=='S':
      n_b+=1
    looped_string += f'L{_}'

  n_l = n_a + n_b + 1


  if not N_tot:
    N_tot = math.ceil((3*n_l) + (7*sse_scale)*(n_b + (X_ab*n_a)))
    
  a_b = ((N_tot - a_l*(n_a + n_b + 1) )/(X_ab*n_a + n_b)) # might be a fraction, convert to int later


  a_a = round(X_ab*a_b)
  a_b = round(a_b)
  motif_lengths = {'L': a_l,
                   'H': a_a,
                   'S': a_b}


  ss_spec = []
  ab_resis = []
  counter = 0
  ab_spec = []
  sse_len = []

  for _ in looped_string:
    ss_spec.append(f'{_},{counter+1}:{counter + motif_lengths[_]}')

    sse_len.append(motif_lengths[_])

    if not _ == 'L':
      ab_resis.append(f'{counter+1}:{counter + motif_lengths[_]}')
      if _ == 'H':
        ab_spec.append(f'H,{counter+1}:{counter + motif_lengths[_]}')

      elif _ == 'S':
        ab_spec.append(f'S,{counter+1}:{counter + motif_lengths[_]}')
    counter += motif_lengths[_]
    

  else:
    _ = 'L'
    end_diff = N_tot-counter

  
  contact_spec = []

  for i in range(len(ori_list)):
    
    resis_i = [int(_) for _ in ab_resis[i].split(':')]

    start_i = resis_i[0]
    stop_i = resis_i[-1]
    _mid_i = math.floor((start_i + stop_i)/2)
    mid_i_ = math.ceil((start_i + stop_i)/2)

    

    if ori_list[i] == '+':
      range_i_1 = f'{start_i}:{_mid_i}'
      range_i_2 = f'{mid_i_}:{stop_i}'
    elif ori_list[i] == '-':
      range_i_1 = f'{mid_i_}:{stop_i}'
      range_i_2 = f'{start_i}:{_mid_i}'

    for j in knn_inds[i]:
      
      resis_j = [int(_) for _ in ab_resis[j].split(':')]

      start_j = resis_j[0]
      stop_j = resis_j[-1]
      _mid_j = math.floor((start_j + stop_j)/2)
      mid_j_ = math.ceil((start_j + stop_j)/2)

      if ori_list[j] == '+':
        range_j_1 = f'{start_j}:{_mid_j}'
        range_j_2 = f'{mid_j_}:{stop_j}'
      elif ori_list[j] == '-':
        range_j_1 = f'{mid_j_}:{stop_j}'
        range_j_2 = f'{start_j}:{_mid_j}'

      contact_spec.append(f'{range_i_1},{range_j_1}')
      contact_spec.append(f'{range_i_2},{range_j_2}')



  return ss_spec, sse_len, contact_spec, N_tot



def parse_contact_spec(contact_spec,chain_len,copies=1):

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


def get_contact_mask(contact_spec,N_tot):
    contact_range_list = parse_contact_spec(contact_spec,N_tot)
    m = jnp.zeros((N_tot,N_tot))
    for (from_i,to_i),(from_j,to_j) in contact_range_list:
        m = m.at[from_i:to_i, from_j:to_j].set(1)
        m = m.at[from_j:to_j, from_i:to_i].set(1)

    return m
