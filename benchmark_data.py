# This module is used to generate different distribution.

import torch

import numpy as np

import torch.utils.data as Data

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

def data_preprocess(X,Y,test_rate=0.2,batch_size=32,shuffle=True,seed=None):
    '''
    Use this function to have a auto np2tensor conversion and splited Torch_loader with loaded data.

    Return: trainloader,train_data,test_data
    '''

    #Transform to torch-tensor and split to trian-test set
    Input=torch.tensor(X,dtype=torch.float32).to('cuda')
    Output=torch.tensor(Y,dtype=torch.float32).to('cuda')

    train_x,test_x,train_y,test_y=train_test_split(Input,Output,test_size=test_rate,random_state=seed,shuffle=shuffle)
    
    #Set Data loader
    train_data  = Data.TensorDataset(train_x, train_y)
    test_data   = Data.TensorDataset(test_x, test_y)
    trainloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    return trainloader,train_data,test_data

class distribution():
    
    def __init__(self,x0=0,x1=1,samples=300):
        '''
        This function can generate the 2D array [x,y] data of three typical distributions,
        including gaussion, uniform, and exponential distribution.

        '''

        self.x=np.linspace(x0,x1,samples)

    def gaussion(self,sigma=.6,mu=.75,A=1,B=0):

        y=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(self.x-mu)**2/(2*sigma**2))

        return np.array([self.x, A*y+B]).T

    def uniform(self,a=.25,b=.75,A=.25,B=.3):

        y=np.where((self.x<a) | (self.x>b), 0, 1/(b-a))

        return np.array([self.x, A*y+B]).T

    def expon(self,m=3,A=.2,B=.3):

        y=np.where(self.x<0, 0, m*np.exp(-m*self.x))

        return np.array([self.x, A*y+B]).T

    def scaler(self,data):
        '''
        Receive the distribution data of 2D array [x,y]

        '''

        data_=MinMaxScaler().fit_transform(data)

        return data_
