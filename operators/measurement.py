# -*- coding: utf-8 -*-

from operations import *
from utils import *
    
class Measurement(nn.Module):
    def __init__(self, density_dim, num_measurements=20):
        super(Measurement, self).__init__()
        self.amp_op = ConstrainedParameter(torch.FloatTensor(num_measurements, density_dim))
        self.amp_op.add_contraint(UnitNormConstraint())
        nn.init.orthogonal_(self.amp_op.data)
        self.pha_op = ConstrainedParameter(torch.FloatTensor(num_measurements, density_dim))
        self.pha_op.add_contraint(RangeConstraint(0, 2 * np.pi))
        nn.init.constant_(self.pha_op.data, 0.0)

    def forward(self, x):
        re_x, im_x = x
    
        re_op, im_op = multiply((self.amp_op, self.pha_op))
        re_op, im_op = density((re_op, im_op))
        
        # only real part is non-zero
        out = torch.matmul(re_x.flatten(-2, -1), re_op.flatten(-2, -1).t()) \
            - torch.matmul(im_x.flatten(-2, -1), im_op.flatten(-2, -1).t())
        
        return out
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    