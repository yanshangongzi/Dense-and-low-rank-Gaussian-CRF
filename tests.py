from layers import ConjugateGradients, dense_gaussian_crf

import torch
import torch.nn as nn
import torch.optim

import unary
import unary1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conjgrad_test(shift=0.01, max_iter=500, tolerance=1e-4, batch_size=10, d=128, H = 50, W = 50):
    solver = ConjugateGradients(shift=shift, max_iter=max_iter, tolerance=tolerance)
    A = torch.normal(mean=0, std=0.1, size=(batch_size, d, H * W), device=device)
    B = torch.normal(mean=0, std=0.1, size=(batch_size, H * W, 1), device=device)
    x_opt = solver.solve(A, B)

    ATA_I = torch.matmul(torch.transpose(A, 1, 2), A) + shift * torch.eye(H * W, device=device)[None, :, :]
    B_opt = torch.matmul(ATA_I, x_opt)

    log_str = "Shift %.7g, tol %.7g, H %d W %d" %(shift, tolerance, H, W)
    assert nn.MSELoss()(B, B_opt) < tolerance, log_str

    del A
    del B
    del x_opt
    del ATA_I
    del B_opt

    torch.cuda.empty_cache()
    print('ConjGrad Success')

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


def run_tests():
    for shift in [1e-3, 1e-2, 1e-1, 1.0]:
        for tolerance in [1e-4, 1e-5, 1e-6]:
            for (H, W) in [(10, 10), (50, 50), (80, 80)]:
                conjgrad_test(shift=shift, tolerance=tolerance, H=H, W=W)

    for shift in [1.0, 1e-1, 1e-2, 1e-3]:
        for (H, W) in [(10, 10), (50, 50), (80, 80)]:
            optimization_test(shift=shift, H=H, W=W)


if __name__ == '__main__':
    run_tests()

