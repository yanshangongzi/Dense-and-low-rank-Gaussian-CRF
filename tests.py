from layers import ConjugateGradients, PottsTypeConjugateGradients, dense_gaussian_crf

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conjgrad_test(shift=0.01, max_iter=500, tolerance=1e-4, batch_size=10, d=128, H = 50, W = 50, L=1):
    solver = ConjugateGradients(shift=shift, max_iter=max_iter, tolerance=tolerance)
    A = torch.normal(mean=0, std=0.1, size=(batch_size, d, H * W * L), device=device)
    B = torch.normal(mean=0, std=0.1, size=(batch_size, H * W * L, 1), device=device)
    x_opt = solver.solve(A, B)

    ATA_I = torch.matmul(torch.transpose(A, 1, 2), A) + shift * torch.eye(H * W, device=device)[None, :, :]
    B_opt = torch.matmul(ATA_I, x_opt)

    log_str = "Shift %.7g, tol %.7g, H %d W %d" %(shift, tolerance, H, W)
    assert nn.MSELoss()(B, B_opt) < tolerance, log_str

    print('ConjGrad Success')


def expand_matrix(A, shift, L):
    res = torch.zeros(A.shape[0], A.shape[2] * L, A.shape[2] * L, device=device)
    ATA = torch.matmul(torch.transpose(A, 1, 2), A)
    for i in range(L - 1):
        for j in range(i + 1, L):
            k = A.shape[2]
            res[:, (i + 1) * k:(i + 2) * k, (j - 1) * k:j * k] = ATA
            res[:, (j - 1) * k:j * k, (i + 1) * k:(i + 2) * k] = ATA

    res += shift * torch.eye(res.shape[1], device=device)[None, :, :]
    return res


def potts_conjgrad_test(shift=0.01, max_iter=500, tolerance=1e-4, batch_size=10, d=128, H = 50, W = 50, L=2):
    solver = PottsTypeConjugateGradients(shift=shift, max_iter=max_iter, tolerance=tolerance)
    A = torch.normal(mean=0, std=0.1, size=(batch_size, d, H * W), device=device)
    B = torch.normal(mean=0, std=0.1, size=(batch_size, H * W * L, 1), device=device)
    x_opt = solver.solve(A, B)

    big_A = expand_matrix(A, shift, L)

    B_opt = torch.matmul(big_A, x_opt)
    log_str = "Shift %.7g, tol %.7g, H %d W %d L %d" %(shift, tolerance, H, W, L)
    assert nn.MSELoss()(B, B_opt) < tolerance * 10, log_str

    print('PottsConjGrad Success')


def optimization_test(batch_size=10, d=128, H = 50, W = 50, n_epochs=200, shift=0.01):
    A = torch.normal(mean=0, std=0.1, size=(batch_size, d, H * W), device=device, requires_grad=True)
    B = torch.normal(mean=0, std=0.1, size=(batch_size, H * W, 1), device=device, requires_grad=True)
    dcrf = dense_gaussian_crf(ConjugateGradients(shift))
    opt = torch.optim.Adam([A, B], lr=3e-3)
    criterion = nn.MSELoss()

    min_loss = torch.tensor(100.0, device=device)
    for i in range(n_epochs):
        opt.zero_grad()
        x = dcrf(A, B)
        loss = criterion(x, torch.zeros_like(x))
        min_loss = min(min_loss, loss)
        loss.backward()
        opt.step()

    assert min_loss < torch.tensor(1e-2 * H, device=device)

    print('CRF Success')


def down_size(in_size):
    out_size = int(in_size)
    out_size = (out_size + 1) // 2
    out_size = int(np.ceil((out_size + 1) / 2.0))
    out_size = (out_size + 1) // 2
    return out_size


def check_unary_prototypes(model, batch_size, n_labels, H, W):
    X = torch.zeros(batch_size, 3, H, W, device=device)
    out = model(X)

    print(out.shape)
    assert out.shape == (batch_size, n_labels, down_size(H), down_size(W))


def check_pairwise_prototypes(model, batch_size, n_labels, H, W, embedding_size):
    X = torch.zeros(batch_size, 3, H, W, device=device)
    out = model(X)

    print(out.shape)
    assert out.shape == (batch_size, embedding_size, n_labels, down_size(H), down_size(W))


def run_layers_tests():
    for shift in [1e-3, 1e-2, 1e-1, 1.0]:
        for tolerance in [1e-4, 1e-5, 1e-6]:
            for (H, W) in [(10, 10), (50, 50), (80, 80)]:
                conjgrad_test(shift=shift, tolerance=tolerance, H=H, W=W)

    for shift in [1.0, 1e-1, 1e-2, 1e-3]:
        for (H, W) in [(10, 10), (50, 50), (80, 80)]:
            optimization_test(shift=shift, H=H, W=W)


if __name__ == '__main__':
    run_layers_tests()
