import torch
import torch.nn.functional as F


def posenet_loss(Pred, GT):
    Pred, GT = Pred.squeeze(dim=1), GT.squeeze(dim=1)
    beta = 500
    loss_G = 0
    loss_weights = [0.3, 0.3, 1]
    l1_criterion = torch.nn.L1Loss()
    mse_criterion = torch.nn.MSELoss()
    # for l, w in enumerate(loss_weights):
    t1, t2 = Pred[:, 0:3], GT[:, 0:3]
    mse_pos = mse_criterion( Pred[:, 0:3] , GT[:, 0:3])
    mse_pos = mse_pos + mse_criterion( F.normalize(t1,p=2,dim=1),F.normalize(t2,p=2,dim=1))
    
    # L2 norm
    pre_norm = F.normalize( Pred[:,3:], p=2, dim=1)
    mse_ori = l1_criterion(Pred[:, 3:]/pre_norm, GT[:,3:])
    loss_G += mse_pos + mse_ori * beta
    return loss_G

def geodesic_loss(Ps, Gs, train_val='train'):
    """ Loss function for training network """
    ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0]) 

    dP = Ps[:,jj] * Ps[:,ii].inv()
    dG = Gs[0][:,jj] * Gs[0][:,ii].inv()
    d = (dG * dP.inv()).log()
    tau, phi = d.split([3,3], dim=-1) 
    geodesic_loss_tr = tau.norm(dim=-1).mean()
    geodesic_loss_rot = phi.norm(dim=-1).mean()

    metrics = {
        train_val+'_geo_loss_tr': (geodesic_loss_tr).detach().item(),
        train_val+'_geo_loss_rot': (geodesic_loss_rot).detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, metrics