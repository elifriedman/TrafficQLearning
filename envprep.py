# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:10:09 2015

@author: friedm3
"""
import networkx as nx
import numpy as np
from kshortest import k_shortest_paths
def t(state): return chr(state+ord('A'))
def getAction(s):
    r = [];
    for state in states[ord(s)-ord('A')]:
        r.append(chr(state+ord('A')))
    return r

def getCost(x,s1,s2):
    i1 = ord(s1)-ord('A')
    i2 = ord(s2)-ord('A')
    try:
        ind = states[i1].index(i2)
        return x.new_costs[i1][ind]
    except ValueError:
        return -1;

states = [
            [1, 2, 3],          # A
            [0, 3, 4],          # B
            [0, 3, 5, 6],       # C
            [0, 1, 2, 4, 6, 7], # D
            [1, 3, 7],          # E
            [2, 6, 8],          # F
            [2, 3, 5, 7, 9, 10],# G
            [3, 4, 6, 10],      # H
            [5, 9, 11],         # I
            [6, 8, 10, 11, 12], # J
            [6, 7, 9, 12],      # K
            [8, 9],             # L
            [9, 10]             # M
        ]
cost_lists = [
            [7, 5, 15],
            [7, 11, 11],
            [5, 7, 11, 9],
            [15, 11, 7, 7, 7, 9],
            [11, 7, 7],
            [11, 9, 13],
            [9, 7, 9, 9, 3, 13],
            [9, 7, 9, 3],
            [13, 9, 2],
            [3, 9, 9, 12, 12],
            [13, 3, 9, 2],
            [2, 12],
            [12, 2]
        ]
costs = {}
g = nx.Graph()
for i in range(len(states)):
  for j in range(len(states[i])):
    costs[(i,states[i][j])] = cost_lists[i][j]
    g.add_edge(i,states[i][j],cost=cost_lists[i][j])

def uniq(pathlist,k=100):
  realpaths = ([],[])
  for i in range(len(pathlist[0])):
    if pathlist[1][i] not in realpaths[1]:
        realpaths[0].append(pathlist[0][i])
        realpaths[1].append(pathlist[1][i])
    if len(realpaths[0])==k:
      break
  return realpaths

def find_subsequence(seq, subseq):
  """
  from: Jaime (http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray)
  """
  target = np.dot(subseq, subseq)
  candidates = np.where(np.correlate(seq,subseq, mode='valid') == target)[0]
    # some of the candidates entries may be false positives, double check
  check = candidates[:, np.newaxis] + np.arange(len(subseq))
  mask = np.all((np.take(seq, check) == subseq), axis=-1)
  return candidates[mask]


K=8
ALpaths = uniq(k_shortest_paths(g,0,11,100,weight='cost'),K)
AMpaths = uniq(k_shortest_paths(g,0,12,100,weight='cost'),K)
BLpaths = uniq(k_shortest_paths(g,1,11,100,weight='cost'),K)
BMpaths = uniq(k_shortest_paths(g,1,12,100,weight='cost'),K)
costs = np.array(ALpaths[0]+AMpaths[0]+BLpaths[0]+BMpaths[0]) # join them
paths = np.array(ALpaths[1]+AMpaths[1]+BLpaths[1]+BMpaths[1])

cost_mat = np.zeros((4*K,4*K))
for i in range(len(cost_mat[0])):
  for j in range(len(cost_mat[0])):
    if i==j:
      cost_mat[i][j] = len(paths[i])
    else:
      path= paths[i]
      for e in range(len(paths[j])-1):
        edge1=paths[j][e:e+2]
        edge2=[edge1[1], edge1[0]]
        if len(find_subsequence(paths[i],edge1))>0 or len(find_subsequence(paths[i],edge2))>0:
          cost_mat[i][j] += 1


def ALfunc(s):
  if s==0: return range(K)
  else: return None
def AMfunc(s):
  if s==0: return range(K,2*K)
  else: return None
def BLfunc(s):
  if s==1: return range(2*K,3*K)
  else: return None
def BMfunc(s):
  if s==1: return range(3*K,4*K)
  else: return None