import numpy as np
import torch
from scipy.interpolate import interp1d

class _MaterialFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wavelength, nk_data, n_interp, k_interp, dl = 0.005) -> torch.Tensor:
        wavelength_np = wavelength.detach().cpu().numpy()

        if wavelength_np < nk_data[0,0]:
            nk_value = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np > nk_data[-1,0]:
            nk_value = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value = n_interp(wavelength_np)+1.j*k_interp(wavelength_np)

        if wavelength_np-dl < nk_data[0,0]:
            nk_value_m = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np-dl > nk_data[-1,0]:
            nk_value_m = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value_m = n_interp(wavelength_np-dl)+1.j*k_interp(wavelength_np-dl)

        if wavelength_np+dl < nk_data[0,0]:
            nk_value_p = nk_data[0,1]+1.j*nk_data[0,2]
        elif wavelength_np+dl > nk_data[-1,0]:
            nk_value_p = nk_data[-1,1]+1.j*nk_data[-1,2]
        else:
            nk_value_p = n_interp(wavelength_np+dl)+1.j*k_interp(wavelength_np+dl)

        ctx.dnk_dl = (nk_value_p - nk_value_m) / (2*dl)
        
        return torch.tensor(nk_value,dtype=torch.complex128 if ((wavelength.dtype is torch.float64) or\
            (wavelength.dtype is torch.complex128)) else torch.complex64, device=wavelength.device)

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        grad = 2*torch.real(torch.conj(grad_output)*ctx.dnk_dl)
        return grad, None, None, None, None
        

class Material():
    def __init__(self, nk_file, dl=0.005, *args, **kwargs):
        f_nk = open(nk_file)
        data = f_nk.readlines()
        f_nk.close()

        nk_data = []
        for i in range(len(data)):
            _line = data[i].split()
            _lamb0 = _line[0]
            if len(_line) == 3:
                _n = _line[1]
                _k = _line[2]
            elif len(_line) == 4:
                _n = _line[2]
                _k = _line[3]
            else:
                raise ValueError("unknown dimensions of refraction data")
            
            nk_data.append([float(_lamb0), float(_n), float(_k)])
        nk_data = np.array(nk_data)

        self.n_interp = interp1d(nk_data[:,0],nk_data[:,1],kind='cubic')
        self.k_interp = interp1d(nk_data[:,0],nk_data[:,2],kind='cubic')

        self.nk_data = nk_data
        self.dl = dl
    
    def apply(self, wavelength, dl = None) -> torch.Tensor:
        if dl is None:
            dl = self.dl

        return _MaterialFn.apply(wavelength, self.nk_data, self.n_interp, self.k_interp, dl) # type: ignore


        