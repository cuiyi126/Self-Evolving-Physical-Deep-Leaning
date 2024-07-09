# In this module, the forward and backward model are defined for the Bp training path.

import torch

def Elayer_forward_fuc(c):
    return electronics_forward(c).simple_io

def Elayer_backward_fuc(c):
    return electronics_backward(c).simple_io

# --Custom electronic layer--

class electronics_forward():

    def __init__(self,c):
        '''
        In this example, a simple transmission function  is used as a real electronic layer which 
        has been evaluated to simplify the demonstration process. 

        In the real test process, the function is an IO program interface for the learning system.

        Electronic hypothesis is defined by the electronic condition.

        '''
        self.h=c 

    def forward_func(self,x):
        if self.h == 0:
            output=0.5*torch.exp(3.5*x-.9)-0.2

        elif self.h == 1:
            output=-0.5*torch.exp(-3.5*x-.9)+0.2

        elif self.h == 2:
            output=8*x**3
            
        return output

    def simple_io(self,X):

        out=self.forward_func(X)

        return out

class electronics_backward():
    
    def __init__(self,c):
        '''
        Backward function of the corresponding electronic layer.

        It may be a differential physical model or a deep neural network model of physical system.

        As an example, we add noise for the  model.
        '''
        self.h=c 

    def backward_func(self,x):
        if self.h == 0:
            output=0.5*torch.exp(3.5*x-.9)-0.2
            output=noise(output)

        elif self.h == 1:
            output=-0.5*torch.exp(-3.5*x-.9)+0.2
            output=noise(output)

        elif self.h == 2:
            output=8*x**3
            output=noise(output)

        return output

    def simple_io(self,X):

        out=self.backward_func(X)

        return out

def noise(pure_output,noise_level=0.01):

    noise = torch.normal(mean=0, std=noise_level, size=pure_output.shape).to('cuda')

    return pure_output+noise

"""class linear():
    def __init__(self,c):
        '''This function could be used to replace the dynamic layer to verify the  difference.
        Use linear(c).io to replace the electronics_forward/backward(c).io
        '''
        self.c=c
    def io(self,x):
        if self.c==0:
            a=x
        elif self.c==1:
            a=2*x
        elif self.c==2:
            a=3*x
        return a
"""
