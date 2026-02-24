import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class MSE_Loss(nn.Module):
    def __init__(self, use_cov_weight=False, prepared_var_mtx=False):
        """
        Args:
            use_cov_weight (bool): 공분산 기반 가중치 적용 여부
            lambda_latent (float): 잠재 공간(Factor) Loss의 반영 비율
            eps (float): 역행렬 계산 시 수치 안정성을 위한 작은 값
        """

        super().__init__()
        self.use_cov_weight = use_cov_weight
        self.prepared_var_mtx = prepared_var_mtx
        self.mse = nn.MSELoss()

    def forward(self, x_hat, x, var_mtx=None):
        """
        Args:
            pred_r: Decoder가 복원한 자산 수익률 (Batch, N, T)
            true_r: 실제 자산 수익률 (Batch, N, T)
        """
        
        if self.use_cov_weight == True and self.prepared_var_mtx == False:
            
            # mse_loss * cov_mtx^-0.5 -> 상대적으로 분산이 작은쪽에서 못맞추는 경우의 loss를 더 키움.(단 다른 자산 간 cov는 0으로 취급)
            
            # 배치 단위로 공분산 행렬 계산이 무거우므로, 
            # 여기서는 '잔차의 분산(Variance)'으로 나누어주는 "Inverse Variance Weighting"을 적용하거나
            # 혹은 단순하게 Mahalanobis Distance의 간소화 버전을 사용합니다.
            
            # [방법 A] 완전한 마할라노비스 (Batch가 클 때만 추천, 연산량 많음)
            # 여기서는 효율성을 위해 [방법 B] Diagonal Weighting (변동성 역가중치) 사용
            
            # 각 자산별 분산(Variance) 계산 (Batch & Time 차원에 대해)
            # var: (1, N, 1) -> Broadcasting 가능하게
            var = torch.var(x, dim=(0, 2), keepdim=True) + 1e-6
            
            # 변동성이 큰 자산의 에러는 좀 깎고, 작은 자산의 에러는 키워줌 (공평하게)
            weighted_diff = (x_hat - x) ** 2 / var
            recon_loss = torch.mean(weighted_diff)


        elif self.use_cov_weight == True and self.prepared_var_mtx == True:
            if var_mtx.dim() != 3:
                var_mtx = var_mtx.view(1, -1, 1)
            
            var_mtx += 1e-6

            weighted_diff = (x_hat - x) ** 2 / var_mtx
            recon_loss = torch.mean(weighted_diff)

           
        else:
            # 그냥 쌩 MSE
            recon_loss = self.mse(x_hat, x)
        
        return recon_loss
    

def entropic_loss(pnl, risk_aversion=1.0):
    # pnl: (batch_size,)
    x = -risk_aversion * pnl
    # log(mean(exp(x))) = logsumexp(x) - log(N)
    # logsumexp는 내부적으로 max 값을 빼서 계산하므로 inf가 안 뜸
    loss = (1/risk_aversion) * (torch.logsumexp(x, dim=0) - torch.log(torch.tensor(x.size(0), device=pnl.device)))
    return loss


def cvar_loss(pnl, alpha=0.05):
    """
    pnl: (Batch Size,) 형태의 텐서. 모델의 헷징 결과로 얻은 최종 손익.
    alpha: 상위 몇 %의 악성 손실을 볼 것인가 (보통 1%, 5%)
    """
    # 1. P&L을 오름차순 정렬 (손실이 큰 순서대로, 즉 값이 작은 순서대로)
    sorted_pnl, _ = torch.sort(pnl)

    # 2. 하위 alpha%에 해당하는 인덱스 계산
    n_samples = pnl.size(0)
    cutoff_index = int((n_samples * alpha // 1))

    # 3. 하위 alpha%의 평균 계산 (Expected Shortfall)
    # pnl이 이익(+)일 수도 손실(-)일 수도 있음.
    # CVaR은 보통 '손실의 크기'를 양수로 표현하므로 -를 붙여줌.
    # 즉, -100원이면 손실이 100원이므로 Loss는 100이 되어야 함.
    tail_loss = -torch.mean(sorted_pnl[:cutoff_index])

    return tail_loss


def KLD_Loss(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if torch.dim(kld) >= 2:
        torch.mean(kld)

    return kld
        