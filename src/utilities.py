from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from itertools import repeat

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    #results_dir = os.environ["RESULTS"]
    results_dir = './results'
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    #results_dir = os.environ["RESULTS"]
    results_dir = './results'
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    #results_dir = os.environ["RESULTS"]
    results_dir = './results'
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int, device='cpu'):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        #yield batch_X.cuda(), batch_y.cuda()
        yield batch_X.to(device), batch_y.to(device)


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    device = next(network.parameters()).device
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size, device):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, P: Tensor = None):
    """Compute a Hessian-vector product.
    
    If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    device = next(network.parameters()).device
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    #vector = vector.cuda()
    vector = vector.to(device)
    if P is not None:
        #vector = vector / P.cuda().sqrt()
        vector = vector / P.to(device).sqrt()
    for (X, y) in iterate_dataset(dataset, physical_batch_size, device):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    if P is not None:
        #hvp = hvp / P.cuda().sqrt()
        hvp = hvp / P.to(device).sqrt()
    return hvp



def compute_block_hvp(network, params, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, P: Tensor = None):
    """Compute a Hessian-vector product.
    
    If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    p = len(parameters_to_vector(params))
    n = len(dataset)
    device = next(network.parameters()).device
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    #vector = vector.cuda()
    vector = vector.to(device)
    if P is not None:
        #vector = vector / P.cuda().sqrt()
        vector = vector / P.to(device).sqrt()
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, params, retain_graph=True)]
        hvp += parameters_to_vector(grads)
        del grads
    if P is not None:
        #hvp = hvp / P.cuda().sqrt()
        hvp = hvp / P.to(device).sqrt()
    return hvp



def lanczos(matrix_vector, dim: int, neigs: int, device='cpu'):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        #gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, P=None):
    """ Compute the leading Hessian eigenvalues.
    
    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    device = next(network.parameters()).device
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs, device=device)
    return evals



def get_block_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=1, physical_batch_size=1000, P=None):
    """ Compute the leading Hessian eigenvalues.
    
    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    device = next(network.parameters()).device
    all_evals = []
    for params in network.parameters():
        layer_hvp_delta = lambda delta: compute_block_hvp(network, params, loss_fn, dataset,
                                              delta, physical_batch_size=physical_batch_size, P=P).detach().cpu()
        nparams = len(parameters_to_vector((params)))
        evals, evecs = lanczos(layer_hvp_delta, nparams, neigs=neigs, device=device)
        all_evals.append(evals)
    return all_evals




def get_signsgd_eos_max(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                    physical_batch_size=1000, n_iter=100, n_restarts=1, P=None):
    """ Compute the EoS quantity of SignGD.
    max_d d^THd/||d||^2_{inf}
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P)
    nparams = len(parameters_to_vector((network.parameters())))
    device = next(network.parameters()).device
    d_t = torch.randn(nparams, dtype=torch.float, device=device)
    eos_max = 0.0
    for _ in range(n_restarts):
        d_t = torch.randn(nparams, dtype=torch.float, device=device)
        d_t = torch.sign(d_t)
        eos_cur = 0.0
        for t in range(1, n_iter+1):
            hvp = hvp_delta(d_t)
            eos_cur = sum(d_t*hvp).detach().item() 
            #d_new = torch.erf(2/np.sqrt(np.pi)*(drct + eta*hvp))
            #d_new = torch.tanh(drct + eta*hvp)
            v_t = torch.sign(hvp)
            gamma_t = 2/(t+3)
            d_t = (1-gamma_t)*d_t + gamma_t*v_t
            wandb.log({
                       "max_d_eos_evolution": eos_cur,
                       #"step_diff_norm": torch.norm(d_new-d_t).detach().item() 
                      })
            
        d_t = torch.sign(d_t)
        hvp = hvp_delta(d_t)
        eos_cur = sum(d_t*hvp).detach().item()
        eos_max = max(eos_cur, eos_max) 
        wandb.log({
                       "max_d_eos_evolution": eos_max,
                       #"step_diff_norm": torch.norm(d_new-d_t).detach().item() 
                      })
    return eos_max, d_t


def project_l12_flat(y: torch.Tensor, model: torch.nn.Module, tau: float = 1.0) -> torch.Tensor:
    """
    Projects the flat vector y onto the mixed l1,2-ball of radius tau, where y is a 1D tensor
    and the block structure is defined by the model's parameters (flattened into y).

    Args:
        y: A 1D torch.Tensor of shape (N,), where N = sum(p.numel() for p in model.parameters()).
        model: A torch.nn.Module whose parameters define the block sizes in y.
        tau: Radius of the l1,2-ball (default=1.0).

    Returns:
        A 1D torch.Tensor of shape (N,) giving the projection of y onto {v: sum_l ||v^{(l)}||_2 <= tau}.
    """
    # Verify flattened size
    shapes = [p.numel() for p in model.parameters()]
    total = sum(shapes)
    if y.ndim != 1 or y.numel() != total:
        raise ValueError(f"y must be a flat vector of length {total}, got shape {tuple(y.shape)}")

    # Split y into blocks matching model.parameters()
    blocks = list(torch.split(y, shapes))

    # Compute block-wise l2 norms
    norms = torch.stack([b.norm(p=2) for b in blocks])

    # If already within the ball, return copy
    if norms.sum() <= tau:
        return y.clone()

    # Sort norms descending, compute cumulative sum
    sorted_norms, idx = torch.sort(norms, descending=True)
    cumsum = torch.cumsum(sorted_norms, dim=0)
    js = torch.arange(1, len(norms)+1, device=y.device, dtype=y.dtype)
    # Compute thresholds
    th = (cumsum - tau) / js
    mask = sorted_norms > th
    # Pick largest k
    k = mask.nonzero(as_tuple=False).max().item()
    mu = th[k]

    # Shrink each block and re-flatten
    proj_blocks = []
    for norm, block in zip(norms, blocks):
        if norm > mu:
            proj_blocks.append((1 - mu / norm) * block)
        else:
            proj_blocks.append(torch.zeros_like(block))
    return torch.cat(proj_blocks)



def get_blocksgd_eos_max(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, left=0, right=0,
                    physical_batch_size=1000, n_iter=100, eta=1, P=None):
    """ 
    Compute the EoS quantity of BlockCD.
    max_d d^THd s.t. ||d||^2_{1,2} <= 1
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P)

    device = next(network.parameters()).device
    nparams = len(parameters_to_vector((network.parameters()))) # number of total parameters
    sizes = [p.numel() for p in network.parameters()] # sizes of each layer
    boundaries = torch.tensor([0] + sizes).cumsum(0).tolist() # start and end indices in the flattened vector of parameters
    eos_max = 0.0
    for _ in range(15):
        #d_t = torch.randn(nparams, dtype=torch.float, device=device)
        d_t = torch.sign(torch.randn(nparams, dtype=torch.float, device=device))
        d_t = project_l12_flat(d_t, network)
        eos_cur = 0.0
        for t in range(1, n_iter//5+1):
            hvp = hvp_delta(d_t)
            eos_cur = sum(d_t*hvp).detach().item() 
            Hd_t_norm_max = 0.0
            l_max = 0
            l = 0
            for c in boundaries[:-1]:
                start, end = boundaries[l], boundaries[l+1]
                cur_norm = torch.sum(torch.mul(hvp[start:end], hvp[start:end]))
                if cur_norm > Hd_t_norm_max:
                    Hd_t_norm_max = cur_norm
                    l_max = l
                l += 1
    
            #print(l_max, Hd_t_norm_max, sizes[l_max])
            #d_new = torch.erf(2/np.sqrt(np.pi)*(drct + eta*hvp))
            #d_new = torch.tanh(drct + eta*hvp)
            v_t = torch.zeros(nparams, dtype=torch.float, device=device)
            start, end = boundaries[l_max], boundaries[l_max+1]
            v_t[start:end] = hvp[start:end]/torch.sqrt(Hd_t_norm_max)
            gamma_t = 2/(t+3)
            d_t = (1-gamma_t)*d_t + gamma_t*v_t
            d_t = project_l12_flat(d_t, network)
            wandb.log({
                       "max_d_eos_evolution": eos_cur,
                       #"step_diff_norm": torch.norm(d_new-d_t).detach().item() 
                      })
            
        hvp = hvp_delta(d_t)
        eos_cur = sum(d_t*hvp).detach().item() 
        wandb.log({
                   "max_d_eos_evolution": eos_cur,
                   #"step_diff_norm": torch.norm(d_new-d_t).detach().item() 
                  })
        eos_max = max(eos_cur, eos_max)

    #d_proj = project_l12_flat(d_t, network)
    #hvp = hvp_delta(d_proj)
    #eos = sum(d_proj*hvp).detach().item()
        
    #wandb.log({
    #    "proj_error": torch.norm(d_t - d_proj).detach().item()
    #})
    
    return eos_max


#@torch.compile  ## I had to comment this out, it was throwing an error
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

#@torch.compile
def PolarExpress(G: torch.Tensor, steps, frob_eps=1e-2, deflation_eps=1e-2):
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):  # opposite convention from our other code
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + frob_eps))

    hs = polar_express_coeffs_list[:steps] + list(repeat(polar_express_coeffs_list[-1], steps - len(polar_express_coeffs_list)))
    for a, b, c in hs:
        a = a / (1 + deflation_eps)
        b = b / (1 + deflation_eps)
        c = c / (1 + deflation_eps)
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def project_inf2_flat(y_flat: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Projects a flattened parameter vector y_flat onto the l_{inf,2}-ball of radius 1,
    where each slice corresponding to a matrix parameter is projected via spectral norm,
    and non-matrix parameters (vectors/scalars) are left unchanged.

    Args:
        y_flat (torch.Tensor): 1-D tensor of shape (P,), where P is total number of params.
        model (torch.nn.Module): network defining parameter shapes and grouping.

    Returns:
        torch.Tensor: projected 1-D tensor of same shape as y_flat.
    """
    # Gather parameter shapes and sizes
    named = list(model.named_parameters())
    shapes = [p.shape for _, p in named]
    sizes = [p.numel() for _, p in named]
    # Compute slice boundaries
    device = y_flat.device
    boundaries = torch.tensor([0] + sizes, device=device).cumsum(0)

    y_proj = torch.empty_like(y_flat)

    for (name, p), start, end, shape in zip(named, boundaries[:-1], boundaries[1:], shapes):
        start, end = int(start), int(end)
        block = y_flat[start:end]
        # If parameter is a matrix or higher (ndim >= 2), project by spectral norm
        if p.ndim >= 2:
            #print(name, p.shape)
            #print(name, shape, p.shape)
            W = block.view(shape[0], -1)
            #W = W.view(W.size(0), -1)
            # Compute spectral norm (largest singular value)
            #sigma = torch.linalg.norm(W, ord=2)
            sigma = torch.linalg.svdvals(W)[0]
            #print(sigma)
            #print(sigma)
            if sigma > 1.0:
                W = W / sigma
            y_proj[start:end] = W.reshape(-1)
        else:
            # Leave vectors/scalars unchanged
            y_proj[start:end] = block

    return y_proj

def get_spectralgd_eos_max(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, left=0, right=0,
                    physical_batch_size=1000, n_iter=100, n_restarts=1, P=None, n_polar_steps=5):
    """ 
    Compute the EoS quantity of SpectralGD.
    max_d d^THd s.t. ||d||^2_{inf,2} <= 1
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P)
    
    nparams = len(parameters_to_vector((network.parameters()))) # number of total parameters
    sizes = [p.numel() for p in network.parameters()] # sizes of each layer
    boundaries = torch.tensor([0] + sizes).cumsum(0) # start and end indices in the flattened vector of parameters
    eos_max = 0.0
    device = next(network.parameters()).device
    for _ in range(n_restarts):
        d_t = torch.randn(nparams, dtype=torch.float, device=device)
        d_t = project_inf2_flat(d_t, network)
        eos_cur = 0.0
        for t in range(1, n_iter+1):
            #d_t = project_inf2_flat(d_t, network)
            hvp = hvp_delta(d_t)
            eos_cur = sum(d_t*hvp).detach().item()
            #eos_max = max(eos_cur, eos_max)
            wandb.log({
                       "max_d_eos_evolution": eos_max
                      })
            
            v_t = torch.zeros(nparams, dtype=torch.float, device=device)
            
            for l, p in enumerate(network.parameters()):
                start, end = boundaries[l].item(), boundaries[l+1].item()
                block = hvp[start:end]
                block = block.view(p.shape)
                # only apply to matrix‐shaped params
                if p.ndim >= 2:
                    # PolarExpress
                    if p.ndim > 2:
                        block = block.view(block.size(0), -1)
                    u = PolarExpress(block, n_polar_steps)
                    v_t[start:end] = u.reshape(-1)

            gamma_t = 2/(t+3)
            d_t = (1-gamma_t)*d_t + gamma_t*v_t


        #d_t = project_inf2_flat(d_t, network)
        hvp = hvp_delta(d_t)
        eos_cur = sum(d_t*hvp).detach().item() 
        eos_max = max(eos_cur, eos_max)
        wandb.log({
                   "max_d_eos_evolution": eos_max
                  })
        

    #d_proj = project_l12_flat(d_t, network)
    #hvp = hvp_delta(d_proj)
    #eos = sum(d_proj*hvp).detach().item()
        
    #wandb.log({
    #    "proj_error": torch.norm(d_t - d_proj).detach().item()
    #})
    
    return eos_max, d_t



def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device=next(network.parameters()).device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0

