import torch

def autograd_fuc(f_forward, f_backward):
    """
    A function that constructs and returns the custom autograd function in Pytorch.
    
    It was used in physics-aware training (PAT, https://www.ncbi.nlm.nih.gov/pubmed/35082422). 
    
    In DPHT for SEA, it's used to execute the error backpropagation path.

    Note that the PAT method is not the only way for training modulation parameters.

    """
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args): 
            ctx.save_for_backward(*args)   #save func_inputs and para
            return f_forward(*args)
        def backward(ctx, grad_output):
            args = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(f_backward, args, v=grad_output)  # [sample_v * J_sample(input)...] N x (1xm dot mxn) Rn--》Rm
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply