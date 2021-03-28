import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def read_txts():
    '''
    read data from data temp folder
    '''
    root = "data/temp/"
    filenames = ["us_removed", "us_population","us_susceptible","us_infected"]
    end = ".txt"
    us_removed = []
    us_population = []
    us_susceptible = []
    us_infected = []
    lis = [us_removed,us_population,us_susceptible,us_infected]
    for file,li in list(zip(filenames,lis)):
        
        li = read_txt(root + file+end)
        print(file+": number of entries: "+str(len(li)))
    #print("US_population:  "+us_population[0])
                              
def read_txt(filename):
    '''
    read text files and outputs the list
    '''
    
    with open(filename, 'r') as f:
        items = f.read().split("\n")
    return items
def sim_fun_ODE(s,infected,beta, N, D, int_steps, length):
    '''
    outputs the predicted susceptible and infected, as two lists
    '''
    S = np.zeros(length)
    I = np.zeros(length)
    S[0] = s[0]
    I[0] = infected[0]
    dt = 1.0/int_steps
    for l in range(length-1):
        for i in range(int_steps):
            S[l] = S[l] - beta*I[l]/N*S[l]*dt
            I[l] = I[l] + (-I[l]/D + beta*I[l]/N*S[l])*dt
        S[l+1] = S[l]
        I[l+1] = I[l]
    return S, I

def sim_fun_SDE(s,i,beta, N, D, int_steps, length):
    '''
    outputs the predicted susceptible and infected, as two lists
    '''
    S = np.zeros(length)
    I = np.zeros(length)
    S[0] = s[0]
    I[0] = i[0]
    dt = 1.0/int_steps
    for l in range(length-1):
        for i in range(int_steps):
            noise_matrix = np.matrix([[beta*I[l]*S[l]/N,-beta*I[l]*S[l]/N],[-beta*I[l]*S[l]/N, beta*I[l]*S[l]/N + I[l]/D]])
            normal_noise = np.matmul(la.sqrtm(noise_matrix), np.random.normal((1,2)))
            S[l] = S[l] - beta*I[l]/N*S[l]*dt + np.sqrt(dt)*normal_noise[0]
            I[l] = I[l] + (-I[l]/D + beta*I[l]/N*S[l])*dt + np.sqrt(dt)*normal_noise[1]
    S[l+1] = S[l]
    I[l+1] = I[l]
    return S, I

def draw_ODE_from_data(output_path,s,i,r,p,beta,d,length,int_steps=1):
    '''
    output_path: where and what's the name of the png file to be outputed example "test/comparison.png"
    int_steps: what is the step between timestamps, in our case is 1 day. 
    length: how long do we want to predict, if 60, then predicting 60 days after the first day in the sequential data. 
    
    '''
    #N = p    # population size
    #beta = betas
    #D = ds
    
    plt.scatter(y=i,x=range(0,len(i),1))
    plt.savefig(output_path)
    S_ODE, I_ODE = sim_fun_ODE(s,i,beta, p, d, int_steps, length)
    S_SDE, I_SDE = sim_fun_SDE(s,i,beta, p, d, int_steps, length)
    plt.plot(I_ODE,label='ODE Infected Predicion')
    #plt.plot(I_SDE,label='SDE result')

    plt.xlabel('Time (days)')
    plt.ylabel('Persons')
    plt.legend()
    #output_path = "test/comparison.png"
    plt.savefig(output_path)
    return output_path


def draw_ODE(path,output_path,beta,d):
    
    s_path = path+"s.txt"
    i_path = path+"i.txt"
    r_path = path+"r.txt"
    p_path = path+"p.txt"
    
    with open(s_path,"r") as f:
        s = f.read().split("\n")
        s = [int(x) for x in s if x!=""]
    with open(i_path,"r") as f:
        i = f.read().split("\n")
        i = [int(x) for x in i if x!=""]
    with open(r_path,"r") as f:
        r = f.read().split("\n")
        r = [int(x) for x in r if x!=""]
        
    with open(p_path,"r") as f:
        p = f.read().split("\n")[0]
        p = int(p)
    try:
        output = draw_ODE_from_data(output_path,s,i,r,p,beta,d,len(s)+20,1)
        print("Successfully drew comparison between model fitted and the actual infection data at this location:")
        print(output)
    except:
        print("Failed to generate plot")
    
    