#Author: Yi Cui
#Date: 11/2023
#Last update: 7/2024

# This is the main program of a simple offline example of distribution conversion by SEA.

import torch

from torch import nn

import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from benchmark_data import data_preprocess,distribution

from electronic_layer import Elayer_backward_fuc,Elayer_forward_fuc

from autograd import autograd_fuc

from evolution import CE_loop

# --PART 1--
'''
Define the electronic layer that is constructed by the reconfigurable electronic device.

It can be customized by rewriting the function in "electronic_layer" module.

'''
def Dynamic_elayer(h):
    '''
    This function is used to reduce the error between experiment and simulation under a selected configuration
    of electronic layer by using PA-BP.
    '''
    return autograd_fuc(Elayer_forward_fuc(h),Elayer_backward_fuc(h))  


# --PART 2--
'''
Define the deep structure for SEA.

The architecture of the deep network is designed based on specific problems.

This example is to demonstrate the distribution conversion task, which is 
the foundation of other complex tasks.

'''

class SEA_Net(nn.Module):
    def __init__(self):
        super(SEA_Net, self).__init__()

        self.elayer=Dynamic_elayer

        '''Define a deep structure with 5 electronic layers'''
        self.m1=AffineLayer(64)
        self.m2=AffineLayer(64)
        self.m3=AffineLayer(64)
        self.m4=AffineLayer(64)
        self.m5=AffineLayer(64)

        self.m6=AffineLayer(64)
        self.rd_w= reduct_w(64,1)
        self.output_fac = nn.Parameter(torch.tensor(0.5).float())

    def forward(self, x, c ):
        '''
        The forward function receives input data and current conditions.
        '''

        x=x.repeat(1,32) # Repeating dimensions of 2-D input
        m1=self.m1(x) 
        s1=self.elayer(c[0])(m1) #L1

        m2=(self.m2(s1))
        s2=self.elayer(c[1])(m2) #L2
        
        m3=(self.m3(s2+s1))
        s3=self.elayer(c[2])(m3) #L3
        
        m4=(self.m4(s3+s2))
        s4=self.elayer(c[3])(m4) #L4
        
        m5=(self.m5(s4+s3)) 
        s5=self.elayer(c[4])(m5) #L5

        m6=(self.m6(s5+s4)) #output 
        out=torch.mm(m6,self.rd_w)*self.output_fac # Linear sum 

        return out

'''
Define linear_adding for the electronic output.

Usually, a matrix multiplication is used as the final layer, but it could be ommited 
especially for simple tasks or the electronic layer with strong computing power.

In this example, the simple but fundamental task can be achieved only by linearly adding the modulated output 
without a matrix mapping layer.

'''
def reduct_w(input_dim,output_dim):
        len=input_dim
        w=torch.zeros((len,output_dim))
        bin=len//output_dim
        for i in range(len):
            for j in range(output_dim):
                class_=i//bin
                if class_ == j: 
                    w[i,j]=torch.ones(1)
        return (w/bin).to('cuda')

'''Define affine_layer for input modulation'''
class AffineLayer(nn.Module):
    def __init__(self,input_feature):
        super(AffineLayer, self).__init__()
        self.A = nn.Parameter(torch.ones(input_feature).normal_(0,1).to('cuda') )
        self.B = nn.Parameter(torch.zeros(input_feature).to('cuda') )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.A.normal_(0, 1)
            self.B.zero_()

    def forward(self, x):
        x=self.A * x + self.B
        x=torch.clamp(x,-0.5,.5)
        return x


# --PART 3--
'''
Define the training loop of Bp path of SEA 

In the inference path, the input data flows to multiplexed forward paths under different electronic conditions.

In the training path, two paths including condition evolution and error backpropagation are optimized alternately.
'''

def Bp_loop(Net,c,loader,epochs=10,lr=5e-3,weight_decay=1e-12):

    '''
    Net: A SEA model to be trained

    loader: The torch data loader

    hyperparameters: epochs  & lr(learning rate) & weight_decay 

    c: conditions of electronic layers

    '''

    learning_rate = lr
    epochs=epochs
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max = epochs  )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net.to(device)
    Loss=[]

    for epoch in range(epochs): 
        for step, (batch_x, batch_y) in enumerate(loader):
            outputs = Net(batch_x,c)
            mse_loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            scheduler.step()
        Loss.append(mse_loss.item())
        print('[{:03d}/{:03d}] MSE Loss: {:.4f}'
        .format(epoch+1, epochs, mse_loss.item()))
    return Loss

def Dynamic_bp_loop(Net,C,Loaders,N_elayers,cycles=20,epochs=5,lr=1e-3,weight_decay=1e-12):
    '''
    This function is used to dynamically train different channels and save the results.

    Net: Electronic SEA model

    C: Conditions of e_layers with multiple dataflows.

    Loaders: A list including different data loaders in pytorch.

    N_elayers: The number of electronic layers.

    cycles: The number of alternating training cycles.

    return 2D array of criteria of channels
    '''

    Epochs=cycles*epochs
    criteria=np.zeros((len(Loaders),Epochs))

    for i in range(cycles):
        for j in range(len(Loaders)):
            criteria[j,i*epochs:(i+1)*epochs]=Bp_loop(Net,C[j*N_elayers:(j+1)*N_elayers],Loaders[j],
            epochs,lr,weight_decay)
    
    return criteria


# --PART 4--
'''
Show final results, including results of condition evolution and final performance of the task.
'''

def Res_show(CE_res,CE_trace,Net,Loaders,N_elayers,cycles=5,epochs=10,lr=1e-3,weight_decay=1e-12,
save_ani=True,save_criteria=True):

    optimized_C=CE_res['Vars'][0]

    history_C=CE_trace['phen_best']

    history_f_best=CE_trace['f_best']

    history_f_avg=CE_trace['f_avg']
    
    '''Show results of condition evoluation'''
    # Evoluation dynamic results
    fig1, ax1 = plt.subplots()

    img_shape=(len(Loaders),N_elayers)

    im = ax1.imshow(history_C[0].reshape(img_shape))
    plt.xlabel('Conditions')
    plt.ylabel('Channels')
    plt.title('Evolution diagram of electronic layer configurations')

    def update(i):
        image_data = history_C[i].reshape(img_shape)
        im.set_array(image_data)
        return [im]

    ani = FuncAnimation(fig1, update, frames=range(len(history_C)), interval=200, blit=True)

    if save_ani == True:
        ani.save('Histoty_Elayers.gif')

    # Fitness over generations
    generations = np.arange(len(history_f_best)) 
    plt.figure(2)
    plt.plot(generations, history_f_best, 'r-', label='Best Score')
    plt.plot(generations, history_f_avg, 'b-', label='Average Score')
    plt.xlabel('Generation')
    plt.ylabel('Scores')
    plt.title('Best And Average Scores Over Generations')
    plt.legend()

    '''Show final results of different data channels'''
    #  Criteria of training

    criteria=Dynamic_bp_loop(Net,optimized_C,Loaders,N_elayers,cycles,epochs,lr,weight_decay)

    if save_criteria == True:
        np.savetxt('DPHT_results.csv',criteria,delimiter=',')

    channels, steps = criteria.shape
    x = np.arange(steps)  

    fig3, axs3 = plt.subplots(channels, sharex=True)

    for i in range(channels):
        axs3[i].plot(x, criteria[i], label='Channel {}'.format(i))
        axs3[i].legend(loc='best')
        axs3[i].set_title('Data channel {}'.format(i))
        axs3[i].set_ylabel('Loss {}'.format(i))

    plt.xlabel('Epoch')
    plt.tight_layout() 

    # Show results on different channels in the same electronic architecture
    channel_results=[]

    for i in range(channels):
        channel_results.append(
            Net(torch.from_numpy(basic_data).to('cuda').to(torch.float32),optimized_C[i*N_elayers:(i+1)*N_elayers])
            .data.cpu().numpy().reshape(1,-1))

    fig4,axs4=plt.subplots(channels)
    
    for i in range(channels):
        axs4[i].plot(basic_data[:,0], True_outputs[i],label='Ideal output {}'.format(i))
        axs4[i].scatter(basic_data[:,0], channel_results[i][0], s=6, c='red',label='Experimental output {}'.format(i))
        axs4[i].legend(loc='best')
        axs4[i].set_title('Data channel {}'.format(i))
        axs4[i].set_ylabel('P(X) {}'.format(i))
        axs4[i].set_xlabel('X')

    plt.show()


# -- PART 5--
'''
Preprare data loaders.
'''
def Preprare_loaders(input,dtr1,dtr2,dtr3,batch_size=32,seed=None):

    batch=batch_size

    # input-->gaussion
    i2g_loader,train_data1,test_data1=data_preprocess(input,dtr1[:,1].reshape(-1,1),seed=seed,batch_size=batch)

    # input-->uniform
    i2u_loader,train_data2,test_data2=data_preprocess(input,dtr2[:,1].reshape(-1,1),seed=seed,batch_size=batch)

    # input-->exponential
    i2e_loader,train_data3,test_data3=data_preprocess(input,dtr3[:,1].reshape(-1,1),seed=seed,batch_size=batch)

    Loaders=[i2g_loader,i2u_loader,i2e_loader]

    return Loaders


# --Main--
if __name__ == '__main__':

    samples=100

    '''Generate different distributions'''
    # Input data distribution
    basic_data=distribution().scaler(distribution(samples=samples).gaussion(0.6,0.5))

    # Target data distributions
    gaussion_data=distribution().scaler(distribution(samples=samples).gaussion())

    uniform_data=distribution().scaler(distribution(samples=samples).uniform())

    expon_data=distribution().scaler(distribution(samples=samples).expon())

    True_outputs=[gaussion_data[:,1],uniform_data[:,1],expon_data[:,1]]

    '''Show different distributions'''
    plt.subplot(4,1,1)
    plt.plot(basic_data[:,0],basic_data[:,1])
    plt.title('Original input')
    plt.subplot(4,1,2)
    plt.plot(gaussion_data[:,0],gaussion_data[:,1])
    plt.title('Gaussion target')
    plt.subplot(4,1,3)
    plt.plot(uniform_data[:,0],uniform_data[:,1])
    plt.title('Uniform target')
    plt.subplot(4,1,4)
    plt.plot(expon_data[:,0],expon_data[:,1])
    plt.title('Exponential target')
    plt.tight_layout()
    plt.show()

    '''Hyperparameters'''
    e_layers=5 # Be fixed after constructed the electronic learning network

    CE_cycles=3 # CE and BP cycles of DPHT  

    CE_epochs=3 

    CE_batch=32

    # After CE/BP, It is possible to continue the BP process after stopping CE to further reduce the loss
    Bp_cycles=10 

    Bp_epochs=1 

    Bp_batch=32

    lr=5e-3

    '''Generate Loaders for multiple targets'''
    CE_Loaders=Preprare_loaders(basic_data,gaussion_data,uniform_data,expon_data,CE_batch)

    BP_Loaders=Preprare_loaders(basic_data,gaussion_data,uniform_data,expon_data,Bp_batch)

    '''Dynamic physical hypothesis training'''
    Net=SEA_Net().to('cuda')

    CE_res,CE_trace=CE_loop(Net,CE_Loaders,e_layers,Bp_loop,cycles=CE_cycles,epochs=CE_epochs) 

    Res_show(CE_res,CE_trace,Net,BP_Loaders,e_layers,cycles=Bp_cycles,epochs=Bp_epochs,lr=lr)













