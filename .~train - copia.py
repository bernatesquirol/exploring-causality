# +
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

# import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math

# import numpy as np
from utils import *
from modules import *
# -

import easydict

default_args = easydict.EasyDict({
    # -----------data parameters ------
        "data_type": 'synthetic',#choices=['synthetic', 'discrete', 'real'],
        "data_filename": 'alarm',
        "data_dir":'data/',
        "data_sample_size":5000, # sample size
        "data_variable_size":10, # num nodes
        "graph_type":'erdos-renyi',
        "graph_degree":3, # expected degree of graph
        "graph_sem_type":'linear-gauss',
        "graph_linear_type":'linear',
        "edge_types":2,
        "x_dims":1,
        "z_dims":1,
        "optimizer":'Adam',
        "graph_threshold": 0.3, # 0.3 is good, 0.2 is error prune
        "tau_A":0.0,
        "lambda_A":0.0,
        "c_A":1,
        "use_A_connect_loss":0,
        "use_A_positiver_loss":0,
    # -----------training hyperparameters   
        "no_cuda": True,
        "seed": 43,
        "epochs": 200,
        "batch_size": 100, # note: should be divisible by sample size, otherwise throw an error
        "lr": 3e-3,# basline rate = 1e-3
        "encoder_hidden": 64,
        "decoder_hidden": 64,
        "temp": 0.5,
        "k_max_iter": 1e2,
        "encoder": 'mlp',
        "decoder": 'mlp',
        "no_factor": False,
        "suffix": '_springs5',
        "encoder_dropout": 0.0,
        "decoder_dropout": 0.0,
        "save_folder": './logs',
        "load_folder":'',
        "h_tol": 1e-8,
        "prediction_steps": 10,
        "lr_decay": 200,
        "gamma": 1.0,
        "skip_first": False,
        "var": 5e-5,
        "hard": False,
        "prior": False,
        "dynamic_graph": False  
    })
default_args.cuda = not default_args.no_cuda and torch.cuda.is_available()
default_args.factor = not default_args.no_factor


def fit_predict(seed=42, data_variable_size=10, graph_degree=3, data_sample_size=5000):
    self = easydict.EasyDict({})
    self.args = default_args
    self.args.seed = seed
    self.args.data_variable_size = data_variable_size
    self.args.graph_degree = graph_degree
    self.args.data_sample_size = data_sample_size
    torch.manual_seed(self.args.seed)
    if self.args.cuda:
        torch.cuda.manual_seed(self.args.seed)

    if self.args.dynamic_graph:
        print("Testing with dynamically re-computed graph.")

    # Save model and meta-data. Always saves in a new sub-folder.
    if self.args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = '{}/exp{}/'.format(self.args.save_folder, timestamp).replace(':','-')
        # safe_name = save_folder.text.replace('/', '_')
        os.makedirs(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'self.args': self.args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")


    # ================================================
    # get data: experiments = {synthetic SEM, ALARM}
    # ================================================
    self.train_loader, self.valid_loader, self.test_loader, self.ground_truth_G = load_data( self.args, self.args.batch_size, self.args.suffix)

    #===================================
    # load modules
    #===================================
    # Generate off-diagonal interaction graph
    off_diag = np.ones([self.args.data_variable_size, self.args.data_variable_size]) - np.eye(self.args.data_variable_size)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    num_nodes = self.args.data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))


    if self.args.encoder == 'mlp':
        self.encoder = MLPEncoder(self.args.data_variable_size * self.args.x_dims, self.args.x_dims, self.args.encoder_hidden,
                             int(self.args.z_dims), adj_A,
                             batch_size = self.args.batch_size,
                             do_prob = self.args.encoder_dropout, factor = self.args.factor).double()
    elif self.args.encoder == 'sem':
        self.encoder = SEMEncoder(self.args.data_variable_size * self.args.x_dims, self.args.encoder_hidden,
                             int(self.args.z_dims), adj_A,
                             batch_size = self.args.batch_size,
                             do_prob = self.args.encoder_dropout, factor = self.args.factor).double()

    if self.args.decoder == 'mlp':
        self.decoder = MLPDecoder(self.args.data_variable_size * self.args.x_dims,
                             self.args.z_dims, self.args.x_dims, self.encoder,
                             data_variable_size = self.args.data_variable_size,
                             batch_size = self.args.batch_size,
                             n_hid=self.args.decoder_hidden,
                             do_prob=self.args.decoder_dropout).double()
    elif self.args.decoder == 'sem':
        self.decoder = SEMDecoder(self.args.data_variable_size * self.args.x_dims,
                             self.args.z_dims, 2, self.encoder,
                             data_variable_size = self.args.data_variable_size,
                             batch_size = self.args.batch_size,
                             n_hid=self.args.decoder_hidden,
                             do_prob=self.args.decoder_dropout).double()

    if self.args.load_folder:
        encoder_file = os.path.join(self.args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(self.args.load_folder, 'decoder.pt')
        self.decoder.load_state_dict(torch.load(decoder_file))

        self.args.save_folder = False

    #===================================
    # set up training parameters
    #===================================
    if self.args.optimizer == 'Adam':
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),lr=self.args.lr)
    elif self.args.optimizer == 'LBFGS':
        self.optimizer = optim.LBFGS(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                               lr=self.args.lr)
    elif self.args.optimizer == 'SGD':
        self.optimizer = optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                               lr=self.args.lr)

    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay,
                                    gamma=self.args.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(self.args.data_variable_size)
    tril_indices = get_tril_offdiag_indices(self.args.data_variable_size)

    if self.args.prior:
        prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
        print("Using prior")
        print(prior)
        log_prior = torch.DoubleTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)

        if self.args.cuda:
            log_prior = log_prior.cuda()

    if self.args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    self.rel_rec = Variable(rel_rec)
    self.rel_send = Variable(rel_send)


    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A*A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0.,0.)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w)-tau)
        return torch.sign(w)*w1


    def update_optimizer(optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr

    #===================================
    # training:
    #===================================

    def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        self.encoder.train()
        self.decoder.train()
        self.scheduler.step()


        # update optimizer
        self.optimizer, lr = update_optimizer(self.optimizer, self.args.lr, c_A)


        for batch_idx, (data, relations) in enumerate(self.train_loader):

            if self.args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data).double(), Variable(relations).double()

            # reshape data
            relations = relations.unsqueeze(2)

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = self.encoder(data, self.rel_rec, self.rel_send)  # logits is of size: [num_sims, z_dims]
            edges = logits

            dec_x, output, adj_A_tilt_decoder = self.decoder(data, edges, self.args.data_variable_size * self.args.x_dims, self.rel_rec, self.rel_send, origin_A, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = self.args.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if self.args.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, self.args.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if self.args.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # compute h(A)
            h_A = _h_A(origin_A, self.args.data_variable_size)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)


            loss.backward()
            loss = self.optimizer.step()

            myA.data = stau(myA.data, self.args.tau_A*lr)

            if torch.sum(origin_A != origin_A):
                print('nan error\n')

            # compute metrics
            graph = origin_A.data.clone().numpy()
            graph[np.abs(graph) < self.args.graph_threshold] = 0

            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            shd_trian.append(shd)

        print(h_A.item())
        nll_val = []
        acc_val = []
        kl_val = []
        mse_val = []

        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t))
        if self.args.save_folder and np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()

        if 'graph' not in vars():
            print('error on assign')


        return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

    #===================================
    # main
    #===================================

    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = self.args.c_A
    lambda_A = self.args.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = self.args.h_tol
    k_max_iter = int(self.args.k_max_iter)
    h_A_old = np.inf

    try:
        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(self.args.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, self.ground_truth_G, lambda_A, c_A, self.optimizer)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, self.args.data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A*=10
                else:
                    break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break


        if self.args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()

        # test()
        print (best_ELBO_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        print(best_NLL_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_NLL_graph))
        print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


        print (best_MSE_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    except KeyboardInterrupt:
        # print the best anway
        print(best_ELBO_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        print(best_NLL_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_NLL_graph))
        print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        print(best_MSE_graph)
        print(nx.to_numpy_array(self.ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(self.ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    return self.ground_truth_G, best_ELBO_graph, best_NLL_graph, best_MSE_graph

ground_truth_G, best_ELBO_graph, best_NLL_graph, best_MSE_graph = fit_predict()

plot_graph(ground_truth_G)

plot_graph(best_ELBO_graph)

self


