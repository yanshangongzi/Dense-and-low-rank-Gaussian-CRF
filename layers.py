import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConjugateGradients: # Модуль для решения СЛУ методом сопряженных градиентов
    def __init__(self, shift=0, max_iter=100, tolerance=1e-3):
        self.shift = shift
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.type = 'default'
        self.epsilon = 1e-15
        self.criterion = nn.MSELoss()


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
            alpha_k = r_kTr_k / (torch.matmul(torch.transpose(p_k, 1, 2), ATAp_k) + self.epsilon)
            x_k += alpha_k * p_k
            r_k_new = r_k - alpha_k * ATAp_k

            if self.criterion(r_k_new, torch.zeros_like(r_k_new)) < self.tolerance:
                break

            beta_k = torch.matmul(torch.transpose(r_k_new, 1, 2), r_k_new) / (r_kTr_k + self.epsilon)
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


class PottsTypeConjugateGradients:
    def __init__(self, shift=0, max_iter=100, tolerance=1e-4):
        self.shift = shift
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.type = 'potts'
        self.epsilon = 1e-15


    def matrix_product(self, A, AT, p, batch_size):
        P = p.reshape(batch_size, -1, A.shape[2])
        Q_1 = torch.matmul(A, torch.transpose(P, 1, 2))

        q_sum_1 = Q_1.sum(dim=-1).unsqueeze(2)
        q_sum_2 = torch.matmul(torch.transpose(A, 1, 2), q_sum_1)

        Q = q_sum_2 + self.shift * torch.transpose(P, 1, 2) - torch.matmul(torch.transpose(A, 1, 2), Q_1)
        return torch.transpose(Q, 1, 2).reshape(batch_size, -1, 1)


    def solve(self, embedding, unary):
        x_0 = torch.normal(mean=0, std=0.1, size=unary.shape, device=device)
        A = embedding
        B = unary
        batch_size = A.shape[0]

        AT = torch.transpose(A, 1, 2)
        x_k = x_0.clone()

        r_k = B - self.matrix_product(A, AT, x_k, batch_size)
        p_k = r_k.clone()

        criterion = nn.MSELoss()
        for k in range(self.max_iter):
            r_kTr_k = torch.matmul(torch.transpose(r_k, 1, 2), r_k)
            ATAp_k = self.matrix_product(A, AT, p_k, batch_size)
            alpha_k = r_kTr_k / (torch.matmul(torch.transpose(p_k, 1, 2), ATAp_k) + self.epsilon)
            x_k += alpha_k * p_k
            r_k_new = r_k - alpha_k * ATAp_k

            if criterion(r_k_new, torch.zeros_like(r_k_new)) < self.tolerance:
                break

            beta_k = torch.matmul(torch.transpose(r_k_new, 1, 2), r_k_new) / (r_kTr_k + self.epsilon)
            p_k = r_k_new + beta_k * p_k
            r_k = r_k_new.clone()

        return x_k

def potts_type_crf(solver):
    class PottsTypeCRF(torch.autograd.Function):

        @staticmethod
        def forward(ctx, embedding, unary):
            x = solver.solve(embedding, unary)
            ctx.save_for_backward(embedding, x)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            A, x = ctx.saved_tensors
            D = A.shape[1]
            P = A.shape[2]
            L = x.shape[1] // P

            AT = torch.transpose(A, 1, 2)
            dL_dx = grad_output
            if len(dL_dx.shape) == 2:
                dL_dx = dL_dx.unsqueeze(0)

            dL_dB = solver.solve(A, dL_dx)

            x_mtx = x.view(-1, L, P)
            dB_mtx = dL_dB.view(-1, L, P)

            x_sum = x_mtx.sum(dim=1)[:, None, :]
            S = x_sum - x_mtx

            dL_dA = torch.matmul(torch.matmul(A, torch.transpose(dB_mtx, 1, 2)), S)
            dL_dA += torch.matmul(torch.matmul(A, torch.transpose(S, 1, 2)), dB_mtx)
            dL_dA = -dL_dA

            return dL_dA / D, dL_dB / D

    return PottsTypeCRF.apply
