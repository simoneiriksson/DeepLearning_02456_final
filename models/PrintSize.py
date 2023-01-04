from torch import nn, Tensor

class PrintSize(nn.Module):
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""
    
    first = True

    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}. ")
            self.first = False
        return x
