
from compressai.ans import BufferedRansEncoder, RansDecoder

from models.pywave import DWT_2D, IDWT_2D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import cast
from models.tcm import TCM


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


class WLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(WLS, self).__init__()
        
        self.dwt = DWT_2D(wave='haar')      
        self.OLP = OLP(in_dim*4,out_dim)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)),dim=2))
    def forward(self,x):
        
        x = self.dwt(x)
        b,_,h,w = x.shape
        x = x.view(b,-1,h*w).permute(0,2,1)
        x = x*torch.exp(self.scaling_factors) 
        x = self.OLP(x)
        return x.view(b,h,w,-1).permute(0,3,1,2)
    
class iWLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(iWLS, self).__init__()
        self.idwt = IDWT_2D(wave='haar')      
        self.OLP = OLP(in_dim,out_dim*4)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)),dim=2))
    def forward(self,x):
        b,_,h,w = x.shape
        x = x.view(b,-1,h*w).permute(0,2,1)
        x = self.OLP(x)
        x = x/torch.exp(self.scaling_factors) 
        x = x.view(b,h,w,-1).permute(0,3,1,2)
        x = self.idwt(x)
        
        return x
        

class OLP(nn.Module):
    
    def __init__(self, in_features, out_dim, bias=True):
        super(OLP, self).__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim
        eye_dim = min(in_features,out_dim)
        self.identity_matrix = torch.eye(eye_dim)  

    def loss(self):
        kernel_matrix = self.linear.weight
        if self.in_dim > self.out_dim:
            gram_matrix = torch.mm(kernel_matrix, kernel_matrix.t())
        else:
            gram_matrix = torch.mm(kernel_matrix.t(), kernel_matrix)
        loss_ortho = F.mse_loss(gram_matrix, self.identity_matrix.to(gram_matrix.device))
        return loss_ortho
            
    def forward(self, x):
        output = self.linear(x)
        
        return output

class TCM_AUXT(TCM):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=64,  M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(config=config,head_dim=head_dim,drop_path_rate=drop_path_rate,N=N,M=M,num_slices=num_slices,max_support_slices=max_support_slices, **kwargs)
       
        self.AuxT_enc = nn.Sequential(
            WLS(3,2*N),
            WLS(2*N,2*N),
            WLS(2*N,2*N),
            WLS(2*N,M),  
        )  
        self.AuxT_dec = nn.Sequential(
            iWLS(M,2*N),
            iWLS(2*N,2*N),
            iWLS(2*N,2*N),
            iWLS(2*N,3),  
        )
    def ortho_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, OLP))
        return cast(Tensor, loss)


    def forward(self, x):
        
        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i,layer in enumerate(self.g_a): 
            y_enc_main = layer(y_enc_main) 
            if i in [0,3,6,9]: #shorcut position for encoder
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main += y_enc_aux
                aux_index += 1
        y = y_enc_main
        
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        
        y_dec_aux  = y_hat
        y_dec_main = y_hat
        
        aux_index = 0
        for i,layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            if i in [2,5,8,9]:   # shorcut position for decoder
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main += y_dec_aux
        x_hat = y_dec_main


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }


    def compress(self, x):
        y_enc_aux = x
        y_enc_main = x
        aux_index = 0
        for i,layer in enumerate(self.g_a): 
            y_enc_main = layer(y_enc_main) 
            if i in [0,3,6,9]: #shorcut position for encoder
                y_enc_aux = self.AuxT_enc[aux_index](y_enc_aux)
                y_enc_main += y_enc_aux
                aux_index += 1
        y = y_enc_main
        y_shape = y.shape[2:]
      
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        
        y_dec_aux  = y_hat
        y_dec_main = y_hat
        
        aux_index = 0
        for i,layer in enumerate(self.g_s):
            y_dec_main = layer(y_dec_main)
            if i in [2,5,8,9]:   # shorcut position for decoder
                y_dec_aux = self.AuxT_dec[aux_index](y_dec_aux)
                aux_index += 1
                y_dec_main += y_dec_aux

        x_hat = y_dec_main.clamp_(0, 1)


        return {"x_hat": x_hat}
