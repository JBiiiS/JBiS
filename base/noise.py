import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion(nn.Module):
    """
    [The Noise Generator]
    원본 주가(R_0)에 노이즈를 섞어서 R_t를 만들어주는 클래스입니다.
    Continuous VP-SDE(Song et al.)의 스케줄을 따릅니다.
    """
    def __init__(self, config):
        super().__init__()
        
        # 설정 가져오기
        self.steps = config.sde_steps   # 예: 1000
        self.beta_min = config.eta_min  # 예: 0.1
        self.beta_max = config.eta_max  # 예: 20.0
        
        # -----------------------------------------------------------
        # 1. Beta Schedule 생성 (Linear Schedule)
        # -----------------------------------------------------------
        # 연속적인 시간 t에서의 beta(t)를 0~1 사이의 1000개 구간으로 쪼갭니다.
        # Discrete Beta_t = Continuous_Beta(t) * (1 / N)
        # 결과값 범위: 0.1/1000(=0.0001) ~ 20/1000(=0.02) -> DDPM 표준 범위와 일치함!
        betas = torch.linspace(self.beta_min / self.steps, 
                               self.beta_max / self.steps, 
                               self.steps)
        
        # 2. Alpha 계산 (Signal Preservation Rate)

        one_minus_betas = 1. - betas
        alphas_squared = torch.cumprod(one_minus_betas, dim=0) # bar_alpha_t
        
        # 3. GPU로 같이 이동하도록 buffer로 등록 (학습 파라미터 아님)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_squared', alphas_squared)
        self.register_buffer('alphas', torch.sqrt(alphas_squared))
        self.register_buffer('h_ts', 1. - alphas_squared)
        self.register_buffer('sqrt_h_ts', torch.sqrt(1. - alphas_squared))

        

    def forward(self, f_0):
        """
        학습 루프에서 호출되는 메인 함수
        Args:
            f_0: (Batch, d, T) - 깨끗한 원본 데이터
        Returns:
            f_t: (Batch, d, T) - 노이즈 낀 데이터
            z: (Batch, d, T) - 정답지 (noise)
            t: (Batch, ) - 랜덤하게 뽑은 시점
        """
        batch_size, device = f_0.shape[0], f_0.device
        
        # 1. 랜덤 시점 t 선택 (0 ~ 999 사이 정수)
        t = torch.randint(0, self.steps, (batch_size,), device=device).long()
        
        # 2. 정답 노이즈 z 생성 (Standard Gaussian)
        z = torch.randn_like(f_0)
        
        # 3. f_t 계산 (Forward Process)
        # 수식: f_t = sqrt(bar_alpha) * f_0 + sqrt(1 - bar_alpha) * noise
        
        # t에 해당하는 계수를 가져옴 (Batch 크기에 맞게)
        alpha = self._extract(self.alphas, t, f_0.shape)
        sqrt_h_t = self._extract(self.sqrt_h_ts, t, f_0.shape)
        
        f_t = alpha * f_0 + sqrt_h_t * z
        
        return f_t, z, t

    def _extract(self, a, t, x_shape):
        """
        배치 인덱스 t에 맞는 값을 배열 a에서 뽑아내고, 
        x_shape에 맞게 차원을 늘려주는 헬퍼 함수 (Broadcasting용)
        Example:
            a: [1000개 값], t: [0, 50, 999] (Batch=3)
            out: (3, 1, 1) 형태의 텐서
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t) 
        
        return out.to(t.device).reshape(batch_size, *((1,) * (len(x_shape) - 1)))