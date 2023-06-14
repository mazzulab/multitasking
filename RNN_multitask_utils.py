# Classes for RNN multitask stimulation and plotting

import torch

# pip install jupyterthemes
##Turn on the dark mode 
from jupyterthemes import jtplot
jtplot.style(theme='monokai')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
from tqdm import tqdm
import os
import pickle

## set a precision type
precision = torch.float32

## check if a GPU available on a Mac
if(torch.backends.mps.is_available()):
  print("GPU available on Mac: mps")
  dev = "mps"
else:
  print("No GPU available... Use CPU.")
  dev = "cpu:0"
device = torch.device(dev)
# torch.backends.mps.empty_cache()



class RecurrentRandomNeuralNetworkNbody:
    """
    Neural network
    replica_n
    activation_phi
    rand_network
    evolve_****
    """
    def __init__(self,neuron_n,replica_n,step_n,time_slice,act_type):
        self.neuron_n   = neuron_n      #a number of neurons 
        self.replica_n  = replica_n    #a number of repricas
        self.step_n     = step_n          #a number of iteration
        self.time_slice = time_slice  # dt in the discretized ODE
        self.act_type   = act_type      #act_type = 1 (tanh), 2(sigmoid), 3(ReLU)
    #activation function
    def activation_phi(self,x):
        if self.act_type == 1:
            return torch.tanh(x)
        if self.act_type == 2:
            return 0.5*torch.tanh(x)+0.5
        if self.act_type == 3:
            y = x
            y[y<=0] = 0
            return y
        else:
            print("choose activation function from {1, 2, 3}.")
            
    def activation_derphi(self,x):
        if self.act_type == 1:
            return 1.0/torch.cosh(x)/torch.cosh(x)
        if self.act_type == 2:
            return 0.5/torch.cosh(x)/torch.cosh(x)
        if self.act_type == 3:
            y = 1
            y[y<=0] = 0
            return y
        else:
            print("choose activation function from {1, 2, 3}.")

    def rand_gauss(self, N, M=1):
        return torch.randn(N,M, dtype=precision, device=device)
    
    def rand_network(self,N,M=1):
        return self.rand_gauss(N,M)
            
    def evolve_rrnn_switch(self, g_scale, J_bias, theta,mu,initial_var, c = 1, sigma_ind = 0, J = [],quench_noise1=[],stim_data=[],task_rule=[]):
        dt = self.time_slice
        N = self.neuron_n
        T = self.step_n
        stim=stim_data['stim']
        n_tasks=len(task_rule["mu"])
        print("running # tasks="+str(n_tasks))
        stim_times=stim_data['stim_times']
        stim_offset=stim_data['stim_offset']
        stim_dur=stim_data['stim_dur']
        nstim=len(stim)
        if stim_data['jitter']: epsilon=0.005*np.random.randn(1000)
        M_ord = torch.zeros(T+1, device=device, dtype=precision)
        C_ord = torch.zeros(T+1, device=device, dtype=precision)
        nrecorded=1000 # stored neurons
        X = torch.zeros((nrecorded, T+1), device=device, dtype=precision)

        if(J==[]):
            J = (g_scale/np.sqrt(N))*self.rand_gauss(N,N)+J_bias*torch.ones(N,N,device=device)/N
        else: print("using J from input")
        for i in range(N):
            J[i][i]=0.0
        sigma = torch.zeros(T+1, device = device, dtype=precision)
        mu_save = torch.zeros(T+1, device = device, dtype=precision)
        x_state = torch.randn(N,device=device,dtype=precision)*initial_var
        X[:nrecorded,0] = x_state[:nrecorded]
        M_ord[0] = torch.mean(x_state)
        C_ord[0] = torch.var(x_state)
        sigma[0]=0.0
        mu_save[0]=0
        if(quench_noise1==[]):
            quench_noise1 = torch.randn(N,device=device,dtype=precision)
            print("generate quenched noise")
        else: print("use quenched noise from input")
        cnt=0
        cnt_task=0
        for i in tqdm(range(T)):
            sigma_task=sigma_ind
            mu_task=mu
            if cnt_task<n_tasks:
                if i>=task_rule["time_onset"][cnt_task]-1:
                    if i<task_rule["time_offset"][cnt_task]:
                        mu_task=task_rule["mu"][cnt_task]
                        sigma_task=task_rule["sigma"][cnt_task]
                    if i==task_rule["time_offset"][cnt_task]: 
                        print("end of task "+str(cnt_task))
                        cnt_task=cnt_task+1 
            sigma0=sigma_task
            mu0=mu_task
            if cnt<=nstim-1:
                if i>stim_times[cnt]:
                    if i<stim_times[cnt]+stim_dur:
                        if stim_data['jitter']: mu0=stim[cnt]*(1+epsilon[cnt])
                        else: mu0=stim[cnt]
                    if i==stim_times[cnt]+stim_dur:
                        print("end of stim"+str(cnt))
                        cnt=cnt+1
                        
                        
            phi_rate = self.activation_phi(c*(x_state- theta))
            x_state =mu0+ (1-dt)*x_state + torch.matmul(J,phi_rate)*dt + sigma0*quench_noise1*dt#+ sigma_flc[i]*quench_noise2*dt+sigma_forget[i]*torch.randn(N,device=device,dtype=precision)*dt#+sigma_p*torch.randn(N,device=device,dtype=precision)*np.sqrt(dt)
            sigma[i] = sigma0
            mu_save[i] = mu0
            M_ord[i+1] = torch.mean(x_state)
            C_ord[i+1] = torch.var(x_state)
            X[:nrecorded,i+1] = x_state[:nrecorded]
        output={'M':M_ord,'C':C_ord,'sigma':sigma,'mu':mu_save,'X':X,"J":J,"quench_noise1":quench_noise1}
        return output
    
    
def plot_time_course(step_n,output,time_slice,pdffigname):

    sigma = output['sigma'].cpu().detach().numpy()
    mu_plot = output['mu'].cpu().detach().numpy()
    Mean = output['M'].cpu().detach().numpy()
    X= output['X'].cpu().detach().numpy()
    Var = output['C'].cpu().detach().numpy()

    t=np.zeros(step_n+1)
    t[0]=0; 
    for i in range(step_n): t[i+1]=t[i]+time_slice

    jtplot.style(theme='grade3')
    cm=plt.get_cmap('Blues') ; cm_interval=[ i /step_n for i in range(1, step_n+1) ] ; cm=cm(cm_interval)
    nplots=4
    fig, ax = plt.subplots(nplots,1, figsize=(30,6))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
    ## Trajectory as a function of time
    ax[0].plot(t,sigma, color="blue", label = "$\sigma(t)$")
    ax[1].plot(t,mu_plot, color="blue", label = "$\mu(t)$")
    ax[2].plot(t,Var, color="red", label = "$C(t,t)$")
    ax[3].plot(t,Mean, color="red", label = "$M(t,t)$")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$\sigma(t)$")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$\mu(t)$")
    ax[2].set_xlabel("$t$") ; ax[2].set_ylabel("C(t,t)")
    ax[3].set_xlabel("$t$"); ax[3].set_ylabel("M(t,t)")
    for i in range(nplots): ax[i].set_xlim([t[0],t[-2]])
    hozon= True

    if hozon == True:

#         pdffigname = "switch_g"+str(g_scale)+"_Jbias"+str(J_bias)+"_c"+str(c)+"_theta"+str(theta)+"_neuron_n"+str(neuron_n)+".pdf"
#         pngfigname = "switch_g"+str(g_scale)+"_Jbias"+str(J_bias)+"_c"+str(c)+"_theta"+str(theta)+"_neuron_n"+str(neuron_n)+".png"
        fig.savefig(pdffigname)
#         fig.savefig(pngfigname)
        print("hozon")
        jtplot.style(theme='monokai')
    else:
        print("Any figure hasn't been saved!")
        jtplot.style(theme='monokai') 

    np.savetxt("temporal_seq_forget.csv", X[0:11,1000:5000].T, delimiter=",")

    fig, ax = plt.subplots(2,1, figsize=(30,6))
    timebins=[i for i in range(len(Var))]
    for i in stim_neurons[0]:
        ax[0].plot(timebins,X[i,timebins], label = "X"+str(i))
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("X")
    ax[0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
    timebins=np.arange(0,np.min([500,step_n]))
    for i in stim_neurons[0]:
        ax[1].plot(timebins,X[i,timebins], label = "X"+str(i))
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("X")
    ax[1].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
    
def plot_time_course_stim(step_n,output,stim_data,time_slice,pdffigname=[]):

    sigma = output['sigma'].cpu().detach().numpy()
    mu_plot = output['mu'].cpu().detach().numpy()
    Mean = output['M'].cpu().detach().numpy()
    X= output['X'].cpu().detach().numpy()
    Var = output['C'].cpu().detach().numpy()
    stim_times=stim_data['stim_times']
    stim_labels=stim_data['stim_labels']
    stim_dur=stim_data['stim_dur']
    
    t=np.zeros(step_n+1)
    t[0]=0; 
    for i in range(step_n): t[i+1]=t[i]+time_slice

    jtplot.style(theme='grade3')
    cm=plt.get_cmap('Blues') ; cm_interval=[ i /step_n for i in range(1, step_n+1) ] ; cm=cm(cm_interval)
    ## 可視化
    nplots=5
    fig, ax = plt.subplots(nplots,1, figsize=(30,10))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
    ## Trajectory as a function of time
    ax[0].plot(t,sigma, color="blue", label = "$\sigma(t)$")
    for i in range(len(stim_times)):
        ax[0].text(stim_times[i],0,str(stim_labels[i]), fontsize=12)    
        ax[0].axvline(stim_times[i],label=(str(stim_labels[i]))) 
        ax[0].axvline(stim_times[i]+stim_dur,linestyle='--',label=(str(stim_labels[i]))) 
    ax[0].legend()
    ax[1].plot(t,mu_plot, color="blue", label = "$\mu$")
    ax[2].plot(t,Var, color="red", label = "$C(t,t)$")
    ax[3].plot(t,Mean, color="red", label = "$M(t,t)$")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$\sigma(t)$")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$\mu(t)$")
    ax[2].set_xlabel("$t$") ; ax[2].set_ylabel("C(t,t)")
    ax[3].set_xlabel("$t$"); ax[3].set_ylabel("M(t,t)")
    timebins=[i for i in range(len(Var))]
    for i in range(5):
        ax[4].plot(timebins,X[i,timebins], label = "X"+str(i))
        ax[4].set_xlabel("$t$") ; ax[4].set_ylabel("X")
    for i in range(len(stim_times)):
        ax[4].text(stim_times[i],0,str(stim_labels[i]), fontsize=12)    
        ax[4].axvline(stim_times[i],label=(str(stim_labels[i]))) 
        ax[4].axvline(stim_times[i]+stim_dur,linestyle='--',label=(str(stim_labels[i]))) 
     
    for i in range(nplots): ax[i].set_xlim([t[0],t[-2]])
    hozon= True

    if hozon == True:
#         pdffigname = "switch_g"+str(g_scale)+"_Jbias"+str(J_bias)+"_c"+str(c)+"_theta"+str(theta)+"_neuron_n"+str(neuron_n)+".pdf"
#         pngfigname = "switch_g"+str(g_scale)+"_Jbias"+str(J_bias)+"_c"+str(c)+"_theta"+str(theta)+"_neuron_n"+str(neuron_n)+".png"
          if len(pdffigname)>0:
                    fig.savefig(pdffigname)
#         fig.savefig(pngfigname)
          print("hozon")
          jtplot.style(theme='monokai')
    else:
        print("Any figure hasn't been saved!")
        jtplot.style(theme='monokai') 

    np.savetxt("temporal_seq_forget.csv", X[0:11,1000:5000].T, delimiter=",")