#!/usr/bin/env python
# coding: utf-8

# # Read and Prepare Data

# In[1]:


import math
import re
import os
import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
from matplotlib import pyplot as plt


# In[2]:


# Reading from txt file and saving it into rows variable as a list of rows
fdir = "dataset-TSPP.xls"
problems = ["eil51","eil76","eil101"]
data = list()
for i in range(3):
    data += [pd.read_excel(fdir,problems[i]).iloc[:,1:5]]


# # Algorithm Classes 

# In[3]:


def dist(x,y = 0):
    return np.sqrt( np.sum((x - y)**2) )


# In[4]:


# Attributes: name, n, coords, profits, DistMat
# Methods: CreateDistMat()
class Problem():
    def __init__(self, Data, PrName, LH = 0):
        typ = "Low" if LH == 0 else "High"
        self.name = f"{PrName}{typ}"
        # Node 0 is duplicated as node n
        self.n = Data.shape[0] + 1
        self.coords = Data.iloc[:,0:2]
        self.coords = pd.concat([self.coords ,self.coords.loc[[0],:]])
        self.profits = Data[typ]
        self.profits = pd.concat([self.profits ,self.profits[0:1]])
        self.DistMat = self.CreateDistMat()
        
    def CreateDistMat(self):
        dist = np.sqrt(            np.power(
            np.diag (self.coords["x"] ) @ np.matrix( self.n*self.n *[1] ).reshape(self.n,-1) -\
            np.matrix( self.n*self.n *[1] ).reshape(self.n,-1) @ np.diag(self.coords["x"] ), 2) +\
            \
            np.power(\
            np.diag (self.coords["y"] ) @ np.matrix( self.n*self.n *[1] ).reshape(self.n,-1) -\
            np.matrix( self.n*self.n *[1] ).reshape(self.n,-1) @ np.diag(self.coords["y"] ), 2)\
                     )
        dist += np.diag(self.n * [float('inf')])
        dist[:,0] = float('inf')
        return np.around(dist,2)
            
    def __repr__(self):
        rep = f"{self.name} TSPP Problem"
        return rep
    
    def __str__(self):
        return self.__repr__()


# In[5]:


# Attributes: prob, T
# Methods: Solve()
class NNsolver():
    def __init__(self, prob):
        self.prob = prob
        self.T = Tour(self.prob)
        
    def Solve(self):
        while not self.T.isComp :
            options = pd.Series(self.T.UpdatedCostMat[self.T[-1],:])
            nxt = options.drop(self.T.nodes).idxmin()
            self.T.add(nxt)
        return self.T
            
    def __repr__(self):
        rep = f"NN Heuristic\nSol:{self.T}"
        return rep
    
    def __str__(self):
        return self.__repr__()


# In[6]:


# Attributes: prob, nodes, isComp, UpdatedCostMat
# Methods: add(), unvisited(), L(), z(), CandList(), LocPherUpd()
class Tour():
    def __init__(self, prob, nodes = [], isComp = False):
        self.prob = prob
        self.isComp = isComp
        
        if type(nodes) in [np.int64,np.int32,int]: nodes=[nodes]
        self.nodes = list(pd.unique(nodes))
        if len(self.nodes) < 1: self.nodes = [0] + self.nodes
        elif self.nodes[0] != 0: self.nodes = [0] + self.nodes
        if isComp and self.nodes[-1] != [self.prob.n-1]:
            self.nodes += [self.prob.n-1]
        
        self.UpdatedCostMat = self.prob.DistMat.copy()
        self.UpdateCostMatrix()
        
    def add(self, newNodes):
        if self.isComp:
            return
        if type(newNodes) in [np.int64,np.int32,int] : newNodes=[newNodes]
        newNodes = list(pd.unique(newNodes))
        newNodes = [node for node in newNodes if node not in self.nodes]
        if newNodes==[]:
            print("No Node can be added!")
            return
        
        self.nodes += newNodes
        if self.prob.n-1 in self.nodes:
            self.isComp = True
            self.__repr__()
        self.UpdateCostMatrix()
        
    def CandList(self, cl):
        i = self.nodes[-1]
        return [j for j in range(self.prob.n)         if self.UpdatedCostMat[i,j] in np.sort(self.UpdatedCostMat[i,:])[:cl]][:cl]
    
    def L(self):
        if len(self.nodes)>1:
            res = np.sum(self.UpdatedCostMat[(self.nodes[:-1],self.nodes[1:])])
        else: res = 0
        return np.round(res,2)
    
    def z(self):
        res = self.L() - np.array(self.prob.profits).sum()
        return np.round(res,2)
    
    def UpdateCostMatrix(self):
        TotalProf = np.array(self.prob.profits).sum()
        EarnedProf = np.array(self.prob.profits)[self.nodes].sum()
        self.UpdatedCostMat[:,-1] = self.prob.DistMat[:,-1] + TotalProf - EarnedProf
    
    def unvisited(self):
        allNodes = list(range(self.prob.n))
        unvisited = [n for n in allNodes if n not in self.nodes]
        return unvisited
    
    def __getitem__(self, key):
        if key < len(self.nodes):
            return self.nodes[key]
        else:
            print("key out of bounds!")
    
    def __repr__(self):
        rep = f"L={self.L()} z={self.z()}"
        if self.isComp:
            rep += f"\nComplete"
        else: rep += f"\nNOT complete"   
        return rep
    
    def __str__(self):
        return self.__repr__()


# In[7]:


# Attributes: T
# Methods: Move(), LocalPhmoneUpd()
class Ant():
    def __init__(self, prob, InitNode):
        self.T = Tour(prob, InitNode)
        
    def Move(self, PhmoneMatrix, q0, alpha, beta, cl, rho, tau0):
        i = self.T[-1]
        opts = set(self.T.CandList(cl)).intersection(self.T.unvisited())
        if opts == set():
            nxt = pd.Series(self.T.UpdatedCostMat[i,:])[self.T.unvisited()].idxmin()
        elif len(opts) == 1:
            nxt = list(opts)
        else:
            visibs = 1/pd.Series(self.T.UpdatedCostMat[i,:]); visibs = visibs[list(opts)]
            phmones = pd.Series(np.array(PhmoneMatrix[i,:])[0]); phmones = phmones[list(opts)]
            RawP = phmones.pow(alpha)*visibs.pow(beta)
            if np.random.random() <= q0:
                nxt = RawP.idxmax()
            else:
                p = RawP/RawP.sum()
                nxt = np.random.choice(list(opts), 1, replace = False,p = p)
        self.T.add(nxt)
        self.LocalPhmoneUpd(PhmoneMatrix, (i,nxt), rho, tau0)

    def LocalPhmoneUpd(self, PhmoneMatrix, arc, rho, tau0):
        PhmoneMatrix[arc] = PhmoneMatrix[arc]*(1-rho) + rho*tau0
    
    def __repr__(self):
        rep = f"Ant traversed {len(self.T.nodes)} nodes.\nL={self.T.L()}"   
        return rep
    
    def __str__(self):
        return self.__repr__()


# In[8]:


# Attributes: prob, ants, phmones, [m, alpha, beta, rho, tau0, q0, cl, MaxIter], Tplus, Lplus
# Members: Solve(), InitGen()
class ACSsolver():
    def __init__(self, prob, params):
        self.m, self.alpha, self.beta, self.rho,        self.tau0, self.q0, self.cl, self.MaxIter = params
        self.prob = prob
        
        self.phmones = np.matrix(self.prob.n*self.prob.n*[self.tau0]).reshape(self.prob.n,-1)
        self.phmones[:,0] = 0
        self.phmones = self.phmones - np.diag(np.diag(self.phmones))
        
        self.ants = list()
        initNodes = np.random.choice(range(1,self.prob.n-1),self.m,replace=False)
        
        for i in range(self.m): self.ants += [Ant(self.prob, initNodes[i])]
        self.Tplus = list()
        self.Lplus = list()
        self.Tplus += [Tour(self.prob, np.random.choice(self.prob.n-1,1) , isComp=True)]
        self.Lplus += [self.Tplus[-1].L()]
        
    def Solve(self):
        itr = 0
        while itr < self.MaxIter:
            itr +=1
            self.MoveAnts()
            self.UpdateTplus()
            self.GlobalPhmoneUpdate()
        print(f"L+={self.Tplus[-1].L()}")
        print(f"z+={self.Tplus[-1].z()}")
        print(f"Visited {len(self.Tplus[-1].nodes)} nodes.")
        
    def MoveAnts(self):
        for ant in self.ants:
            for step in range(1,self.prob.n-1):
                if ant.T.isComp == False:
                    ant.Move(self.phmones, self.q0, self.alpha, self.beta, self.cl, self.rho, self.tau0)
                        
    def UpdateTplus(self):
        TplusNew = self.Tplus[-1]
        for ant in self.ants:
            if ant.T.L() < self.Lplus[-1]:
                TplusNew = ant.T
        self.Tplus += [TplusNew]
        self.Lplus += [TplusNew.L()]
        
    def GlobalPhmoneUpdate(self):
        delta = 1/self.Lplus[-1]
        for idx in range(len(self.Tplus[-1].nodes[:-1])):
            i=self.Tplus[-1].nodes[idx]
            j=self.Tplus[-1].nodes[idx+1]
            self.phmones[i,j] = self.phmones[i,j]*(1-self.rho) + delta*self.rho
    
    def __repr__(self):
        rep = f"ACS Solver\nzPlus={self.Tplus[-1].z()}"
        return rep
    
    def __str__(self):
        return self.__repr__()


# # Running Algorithm 

# In[9]:


prList = list()
for i in range(3):
    prList += [Problem(data[i],problems[i],0)]
    prList += [Problem(data[i],problems[i],1)]


# In[10]:


LNN = list()
tau0 = list()
for p in prList:
    nnObj = NNsolver(p)
    nnObj.Solve()
    LNN += [np.round(nnObj.T.L(),2)]
    tau0 += [1/(p.n*LNN[-1])]


# In[11]:


acsObj = list()


# ## eil51 Low Demand

# In[12]:


probNo = 0


# In[13]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [50,1,2,0.1,tau0[probNo],0.9,15,500]


# In[14]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## eil51 High Demand

# In[15]:


probNo+=1


# In[16]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [50,1,2,0.1,tau0[probNo],0.9,15,500]


# In[17]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## eil76 Low Demand

# In[18]:


probNo+=1


# In[19]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [75,1,2,0.1,tau0[probNo],0.9,15,500]


# In[20]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## eil76 High Demand

# In[21]:


probNo+=1


# In[22]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [75,1,2,0.1,tau0[3],0.9,15,500]


# In[23]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## eil101 Low Demand

# In[24]:


probNo+=1


# In[25]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [100,1,2,0.1,tau0[0],0.9,15,500]


# In[26]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## eil101 High Demand

# In[27]:


probNo+=1


# In[28]:


#params=[m, alpha, beta, rho, tau0, q0, cl, MaxIter]
params = [100,1,2,0.1,tau0[0],0.9,15,500]


# In[29]:


tstart = time.process_time_ns()
np.random.seed(13)
acsObj += [ACSsolver(prList[probNo],params)]
acsObj[-1].Solve()
tend = time.process_time_ns()
print(f"Executed in {1.e-9*(tend - tstart)} CPU*seconds.")


# ## Solutions 

# In[30]:


for solver in acsObj:
    print(solver.Tplus[-1].nodes)

