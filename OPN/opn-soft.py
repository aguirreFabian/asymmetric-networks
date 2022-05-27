#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:03:16 2021

@author: fabian

This program samples from the multidimensional distribution correponding to the
soft version of Gardner problem with fixed asymmetry.

It needs as an input a file named "specs.txt", which should have the next variables in consecutive lines:

    N (int) number of nodes of the graph
    c (int) degree of the graph
    lam (double) level of asymmetry imposed by lambda
    kr (double) margin of the constraints, before multiplication by sqrt(c)
    T2 (int) total number of MCMC steps after annealing
    rate1 (int) number of MCMC steps between samples

It produces the next output:

    samples (dir) it creates a directory with different samples
    opn_dynamics.txt (file) a file with the time series for energy, acceptance rate, eta (asymmetry), unsat fraction
    params.txt (file) a file with the parameters used in the MCMC simulation

"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy import special
import os

def eta_fun(J):
    eta_n =0
    eta_d =0
    for e in J.edges:
        eta_n+= 2*J[e[0]][e[1]]["0"]*J[e[0]][e[1]]["1"]
        eta_d+= J[e[0]][e[1]]["0"]**2  + J[e[0]][e[1]]["1"]**2
    return np.float(eta_n/eta_d)


def asymmetric_J_regular(lam,N,c):
    S = np.matrix([[-1,1],[1,1]])
    S0 = np.matrix([[-.5,.5],[.5,.5]])
    A = np.matrix([[np.sqrt(1-lam),0],[0,np.sqrt(1+lam)]])
    B = S@A@S0
    J = nx.random_regular_graph(c,N)
    J = nx.relabel.convert_node_labels_to_integers(J)
    for e in J.edges:
        w = B@np.random.normal(size=(2,1))
        J[e[0]][e[1]]["0"] = np.float(w[0]/np.sqrt(2))
        J[e[0]][e[1]]["1"] = np.float(w[1]/np.sqrt(2))
    return J

def check_opn(J,kappa):
    found = 1
    rows=list()
    unsat = 0
    for v in J.nodes():
        s = 0
        for e in J.edges(v):
            if v<e[1]:
                s += J[e[0]][e[1]]["0"]
            else:
                s += J[e[0]][e[1]]["1"]
        if s<kappa:
            found = 0
            unsat +=1
            rows.append(1)
        else:
            rows.append(0)
    return found,rows,unsat

def check_opn_node(J, kappa, shift, v):
    s = shift
    for e in J.edges(v):
        if v<e[1]:
            s += J[e[0]][e[1]]["0"]
        else:
            s += J[e[0]][e[1]]["1"]
    if s<kappa:
        unsat = 1
    else:
        unsat = 0
    return unsat
    

def force_opn(J,kappa, eps = 1e-3):
    found, rows, unsat = check_opn(J, kappa)
    for v in J.nodes:
        if rows[v] == 1:
            u = list(J.edges(v))[0][1]
            s = 0
            for e in J.edges(v):
                if not e[1] == u:
                    if e[0]<e[1]:
                        s += J[e[0]][e[1]]["0"]
                    else:
                        s += J[e[0]][e[1]]["1"]
            if v<u:
                J[v][u]["0"] = kappa - s + eps
            else:
                J[v][u]["1"] = kappa - s + eps
    return J
            

def H(x):
    return special.erfc(x/np.sqrt(2))/2

def Hamiltonian0(J,lam,beta):
    energy = 0
    invlam = 1/(1-lam**2)
    for e in J.edges:
        x = J[e[0]][e[1]]["0"]
        y = J[e[0]][e[1]]["1"]
        energy += -.5*invlam*beta*(x**2 + y**2 - 2*lam*x*y)
    return energy

def constraints(J,kappa,gamma):
    energy = 0
    for v in J.nodes():
        s = 0
        for e in J.edges(v):
            if v<e[1]:
                s += J[e[0]][e[1]]["0"]
            else:
                s += J[e[0]][e[1]]["1"]
        if s-kappa<0:
            energy += -gamma*np.abs(s-kappa)**2
    return energy

def diffGaussian(J,lam,e,choice,shift,beta):
    invlam = 1/(1-lam**2)
    x = J[e[0]][e[1]]["0"]
    y = J[e[0]][e[1]]["1"]
    energy = -.5*invlam*(x**2 + y**2 - 2*lam*x*y)
    if choice<.5:
        x = x + shift
    else:
        y = y + shift
    return beta*(-.5*invlam*(x**2 + y**2 - 2*lam*x*y) - energy)

def diffConstraints(J,kappa,gamma,edge,choice,shift):
    if choice<.5:
        v = np.min(edge)
    else:
        v = np.max(edge)
    s = 0
    snew = 0
    for e in J.edges(v):
        if v<e[1]:
            s += J[e[0]][e[1]]["0"]
            if e[1] == edge[1]:
                snew += J[e[0]][e[1]]["0"] + shift
            else:
                snew += J[e[0]][e[1]]["0"]
        else:
            s += J[e[0]][e[1]]["1"]
            if e[1] == edge[1]:
                snew += J[e[0]][e[1]]["1"] + shift
            else:
                snew += J[e[0]][e[1]]["1"]
    hnew = snew-kappa
    h = s-kappa
    if hnew >=0 and h>=0:
        return 0
    if hnew >=0 and h<0:
        return gamma*np.abs(h)**2
    if hnew<0 and h>=0:
        return -gamma*np.abs(hnew)**2
    if hnew<0 and h<0:
        return -gamma*np.abs(hnew)**2 + gamma*np.abs(h)**2


    
if __name__ == '__main__':
    tic = time.perf_counter()
    # Parameters
    
    f = open("specs.txt")
    N = int(f.readline())
    c = int(f.readline())
    lam = float(f.readline())
    kr = float(f.readline())
    T2 = int(f.readline())
    rate1 = int(f.readline())
    f.close()
    
    kappa = kr * np.sqrt(c)

    # Initialize
    J = asymmetric_J_regular(lam, N, c)
    
    L = J.number_of_edges()
    # Simulation params
    # Annealing rounds
    P = 40
    gammas = 20*np.ones(P)
    gammas[0:10] = np.linspace(1,20,10)
    delta = 1
    delta2 = 10
    
    energy = Hamiltonian0(J,lam,1) + constraints(J, kappa, gammas[0])
    energylist = list()
    energylist.append(energy)
    
    etalist = list()
    unsatlist = list()

    
    accrates = list()
    count = 0
    accepted = 0
    samples = 0
    T = 100
    # T2 = int(1e5)
    rate0 = 50
    # rate1 = int(1e3) 
    ratesave = int(rate1/5)

    f = open("params.txt","w")
    f.write(f"N = {N}\n")
    f.write(f"c = {c}\n")
    f.write(f"lam = {lam}\n")
    f.write(f"kappa = {kr}\n")
    f.write(f"P = {P}\n")
    f.write(f"T = {T}\n")
    f.write(f"T2 = {T2}\n")
    f.write(f"delta = {delta}\n")
    f.write(f"delta2 = {delta2}\n")
    f.close()
    
    os.system("mkdir samples")

    for i,gamma in enumerate(gammas):
        for t in range(T):
            count +=1
            index = np.random.randint(L)
            edge = list(J.edges)[index]
            shift = delta*np.random.normal()
            choice  = np.random.rand()
            energydiff = diffGaussian(J, lam, edge, choice, shift, 1) + diffConstraints(J, kappa, gamma, edge, choice, shift)
            prob = np.min([1,np.exp(energydiff)])
            if np.random.rand()<prob:
                accepted +=1
                if choice<.5:
                    J[edge[0]][edge[1]]["0"] = J[edge[0]][edge[1]]["0"] + shift
                else:
                    J[edge[0]][edge[1]]["1"] = J[edge[0]][edge[1]]["1"] + shift
                energy = energy + energydiff
            if np.mod(t,rate0) == 0:
                energylist.append(energy)
                etalist.append(eta_fun(J))
                unsatlist.append(check_opn(J, kappa)[2]/N)
                accrates.append(accepted/count)
    
    J = force_opn(J, kappa)
    energy = Hamiltonian0(J, lam, 1)
    for t in range(T2):
        count +=1
        index = np.random.randint(L)
        edge = list(J.edges)[index]
        shift = delta2*np.random.normal()
        choice  = np.random.rand()
        energydiff = diffGaussian(J, lam, edge, choice, shift, 1)
        prob = np.min([1,np.exp(energydiff)])
        if choice < 0.5:
            unsat = check_opn_node(J, kappa, shift, edge[0])
        else:
            unsat = check_opn_node(J, kappa, shift, edge[1])
        if unsat == 1:
            prob = -1
        if np.random.rand()<prob:
            accepted += 1
            if choice<.5:
                J[edge[0]][edge[1]]["0"] = J[edge[0]][edge[1]]["0"] + shift
            else:
                J[edge[0]][edge[1]]["1"] = J[edge[0]][edge[1]]["1"] + shift
            energy = energy + energydiff
        if np.mod(t,ratesave) == 0:
            energylist.append(energy)
            etalist.append(eta_fun(J))
            unsatlist.append(check_opn(J, kappa)[2]/N)
            accrates.append(accepted/count)
        if np.mod(t,rate1) == 0:
            samples +=1
            Js = list(J.edges.data())
            f = open(f"samples/sample_{samples:03d}.txt","w")
            f.write(str(Js))
            f.close()
        
    
    f = open("opn_dynamics.txt","w")
    f.write("energy,accrate,eta,unsat\n")
    for i in range(len(etalist)):
        f.write(f"{energylist[i]},{accrates[i]},{etalist[i]},{unsatlist[i]}\n")
    f.close()

    toc = time.perf_counter()
    f = open("params.txt","a")
    f.write(f"time  = {toc - tic}\n")
    f.close()

        
                
                
                
        
