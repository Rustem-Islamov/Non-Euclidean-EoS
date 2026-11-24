from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector

import argparse

from archs import load_architecture
from utilities import get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, get_signsgd_eos_max, compute_hvp
from data import load_dataset, take_first, DATASETS
from methods import MySignSGD

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb


def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, fw_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, args = None):
    #results_dir = os.environ["RESULTS"]
    results_dir = './results'
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/signgd/lr_{lr}_beta_{beta}"
    print(f"output directory: {directory}")

    device = f'cuda:{args.device}'

    wandb.init(
            project='EoS_SignlGD_gradnorm_quadratic',
            name = f'muon_{args.arch_id}_{args.dataset}_{args.lr}_{args.n_fw_restarts}_{args.n_fw_steps}',
            config={
            "dataset":dataset,
            "arch_id": arch_id,
            "loss": loss,
            "opt": opt,
            "lr": lr,
            "device": device,
            "max_steps": max_steps,
            "physical_batch_size": physical_batch_size,
            "beta": beta,
            "seed":seed,
            "n_fw_restarts": args.n_fw_restarts,
            "n_fw_steps": args.n_fw_steps
            }
        )
    
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    #network = load_architecture(arch_id, dataset).cuda()
    network = load_architecture(arch_id, dataset).to(device)

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    #optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)
    optimizer = MySignSGD(network.parameters(), lr, beta, eig_freq)

    hvp_delta = lambda delta: compute_hvp(network, loss_fn, abridged_train,
                                              delta, physical_batch_size=physical_batch_size)

    train_loss, test_loss, train_acc, test_acc, direct_smooth_abs, direct_smooth, grad_norm = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eos_max_d = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0)
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    optimizer.zero_grad()

    for step in range(0, max_steps):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        direct_smooth_abs[step] = optimizer.direct_smooth_abs
        direct_smooth[step] = optimizer.direct_smooth
        grad_norm[step] = optimizer.prev_grad_norm_2

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)

            print("eigenvalues: ", eigs[step//eig_freq, :])
            wandb.log({"step":step,
                   "eig1": eigs[step//eig_freq, 0],
                   "eig2": eigs[step//eig_freq, 1],
                   "eig_comp_step": step,
                  })
        if fw_freq != -1 and step % fw_freq == 0:
            if step % (10*fw_freq) == 0:
                eos_FW, d_t = get_signsgd_eos_max(network, loss_fn, abridged_train, physical_batch_size=physical_batch_size, n_iter=args.n_fw_steps, n_restarts=args.n_fw_restarts)
            else:
                eos_FW, _ = get_signsgd_eos_max(network, loss_fn, abridged_train, physical_batch_size=physical_batch_size, n_iter=args.n_fw_steps, n_restarts=args.n_fw_restarts)
                
            wandb.log({
                   "max_d_eos_FW": eos_FW,
                   "spectral_step": step
            })

            wandb.log({
                   "d_t_eos_FW": sum(hvp_delta(d_t)*d_t).item(),
                   "d_t_step": step
            })
            

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("max_d_eos", eos_max_d[:step // eig_freq]), ("grad_norm", grad_norm[:step]), 
                                   ("dir_smooth", direct_smooth[:step]), ("dir_smooth_abs", direct_smooth_abs[:step]), 
                                   ("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}\t{direct_smooth[step]:.3f}")
        wandb.log({"step":step, 
                   "train_loss": train_loss[step], 
                   "train_acc": train_acc[step],
                   "test_loss": test_loss[step],
                   "test_acc": test_acc[step],
                   "direct_smooth": direct_smooth[step],
                   "direct_smooth_abs": direct_smooth_abs[step],
                   "grad_norm": grad_norm[step]
                  })

        #if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
        #    break

        
        total_loss = 0.
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            #loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
            total_loss += loss.item()
            loss.backward(create_graph=True)
        nparams = len(parameters_to_vector(network.parameters()))
        sizes = [p.numel() for p in network.parameters()]
        boundaries = torch.tensor([0] + sizes).cumsum(0)
        w, update, full_grad, _ = optimizer.step(cur_loss=total_loss, nparams=nparams, boundaries=boundaries, device=device)
        ######### compute alignment of the update and LMO(H(w_t)*update) ############

        if fw_freq != -1 and step % (10*fw_freq) == 0:
            for restart in range(5):
                u_t = torch.randn(nparams, dtype=torch.float, device=device)
                u_t = torch.sign(u_t)
                for t in range(1, args.n_fw_steps+1):
                    hvp = hvp_delta(update)
                    v_t = torch.sign(u_t*hvp)

                    gamma_t = 2/(t+3)
                    u_t = (1-gamma_t)*u_t + gamma_t*v_t

            wandb.log({"cos_similarity_update_and_LMO": sum(update*u_t).item()/torch.norm(update).item()/torch.norm(u_t).item(),
                       "cos_similarity_step": 5*step+restart})
            
        if fw_freq != -1 and step % (10*fw_freq) == 0:
            Delta = torch.zeros(nparams, dtype=torch.float, device=device)
            

            for quadratic_step in range(50):
                hvp = hvp_delta(Delta)
                quadratic_grad = full_grad + hvp
                quadratic_loss = total_loss + sum(full_grad*Delta).item() + 0.5*sum(Delta*hvp).item()
                wandb.log({"quadratic_loss": quadratic_loss,
                           "quad_step": step+quadratic_step})
                Delta -= lr*torch.sum(torch.abs(quadratic_grad))*torch.sign(quadratic_grad)
            
        #optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        

    save_files_final(directory,
                     [("dir_smooth", direct_smooth[:step + 1]), 
                      ("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using sign gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("--lr", type=float, help="the learning rate")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="signgd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--n_fw_steps", type=int,
                        help="number of Frank-Wolfe iterations", default=50)
    parser.add_argument("--n_fw_restarts", type=int,
                        help="number of restarts of Frank-Wolfe", default=1)
    parser.add_argument("--device", type=int,
                        help="cuda number", default=0)
    parser.add_argument("--fw_freq", type=int, default=-1,
                        help="the frequency at which we run Frank-Wolfe algorithm (-1 means never)")
    
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)", default=0.0)
    parser.add_argument("--batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=64)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq, fw_freq=args.fw_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, args=args)
