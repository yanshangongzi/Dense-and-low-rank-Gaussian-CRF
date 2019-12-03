import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConjugateGradients: # Модуль для решения СЛУ методом сопряженных градиентов
    def __init__(self, shift=0, max_iter=500, tolerance=1e-3):
        self.shift = shift
        self.max_iter = max_iter
        self.tolerance = tolerance

    def matrix_product(self, A, AT, x): # вычисление матричного произведения 
        A_x = torch.matmul(A, x)        # для матрицы вида A^T * A + lambda * I
        ATA_x = torch.matmul(AT, A_x) + self.shift * x
        return ATA_x

    def solve(self, embedding, unary):
        x_0 = torch.normal(mean=0, std=0.1, size=unary.shape, device=device)
        A = embedding
        B = unary

        AT = torch.transpose(A, 1, 2)
        x_k = x_0.clone()

        r_k = B - self.matrix_product(A, AT, x_k)
        p_k = r_k.clone()

        for k in range(self.max_iter):
            r_kTr_k = torch.matmul(torch.transpose(r_k, 1, 2), r_k)
            ATAp_k = self.matrix_product(A, AT, p_k)
            alpha_k = r_kTr_k / torch.matmul(torch.transpose(p_k, 1, 2), ATAp_k)
            x_k += alpha_k * p_k
            r_k_new = r_k - alpha_k * ATAp_k

            if torch.max(r_k_new) < self.tolerance:
                break

            beta_k = torch.matmul(torch.transpose(r_k_new, 1, 2), r_k_new) / r_kTr_k
            p_k = r_k_new + beta_k * p_k
            r_k = r_k_new.clone()

        return x_k


def dense_gaussian_crf(solver):
    class DenseGaussianCRF(torch.autograd.Function):

        @staticmethod
        def forward(ctx, embedding, unary):
            x = solver.solve(embedding, unary)
            ctx.save_for_backward(embedding, x)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            A, x = ctx.saved_tensors
            AT = torch.transpose(A, 1, 2)
            dL_dx = grad_output
            if len(dL_dx.shape) == 2:
                dL_dx = dL_dx.unsqueeze(0)

            dL_dB = solver.solve(A, dL_dx)

            dL_dA = torch.matmul(torch.matmul(A, dL_dB), torch.transpose(x, 1, 2))
            dL_dA += torch.matmul(torch.matmul(A, x), torch.transpose(dL_dB, 1, 2))
            dL_dA = -dL_dA

            return dL_dA, dL_dB

    return DenseGaussianCRF.apply
