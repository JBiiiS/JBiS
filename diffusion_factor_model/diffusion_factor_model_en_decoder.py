import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, factor_dim, dropout=0.1):
        super().__init__()
        # 정규화(Norm) 층을 완전히 제거했습니다. (No BatchNorm, No LayerNorm)
        self.conv1 = nn.Conv1d(factor_dim, factor_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(factor_dim, factor_dim, 3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, channels, T)
        # 순수하게 컨볼루션과 활성화 함수만 통과
        h = self.conv1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 입력값 x를 그대로 더해주어(Residual) 정보 흐름을 원활하게 하고 학습을 돕습니다.
        return x + h

class Encoder(nn.Module):
    def __init__(self, input_dim, factor_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. 차원 축소 (Compression)
        # 자산(input_dim)들을 선형 결합하여 팩터(feature_dim)의 초안을 만듭니다.
        # 1x1 Conv는 수학적으로 Wx와 동일하므로, 초기 팩터 로딩 행렬(Beta) 역할을 합니다.
        self.start_conv = nn.Conv1d(input_dim , factor_dim, 1)
        
        # 2. 비선형 특징 추출 (Refinement)
        # 단순 선형 결합으로는 잡지 못하는 시계열적 패턴(추세, 국면)을 반영합니다.
        self.blocks = nn.ModuleList([
            ResBlock(factor_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: (Batch, input_dim, T)
        
        f_0 = self.start_conv(x)
        
        for block in self.blocks:
            f_0 = block(f_0)
            
        return f_0 # F_0: (Batch, factor_dim, T)
    

class Decoder(nn.Module):
    def __init__(self, factor_dim, output_dim, num_layers=2, dropout=0.1):
        """
        Args:
            feature_dim: 팩터 차원 (Encoder/Dynamics의 출력 채널 수)
            output_dim: 원래 자산 개수 (input_dim과 동일)
            num_layers: 팩터 정제를 위한 ResBlock 개수
        """
        super().__init__()
        
        # 1. 팩터 상호작용 및 정제 (Refinement)
        # 팩터가 바로 자산으로 가기 전에, 팩터들끼리의 비선형적인 관계를 한번 더 정리합니다.
        # Encoder와 대칭되게 ResBlock을 사용하며, 역시 Normalization은 뺍니다.
        self.blocks = nn.ModuleList([
            ResBlock(factor_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 2. 자산 공간으로 확장 (Projection)
        # (Batch, factor_dim, T) -> (Batch, output_dim, T)
        # 수학적으로 R = Beta * F 와 동일한 역할을 수행합니다.
        # Kernel size=1을 사용하여, '매 시점마다' 팩터들의 선형 결합으로 자산 가격을 계산합니다.
        self.final_conv = nn.Conv1d(factor_dim, output_dim, 1)
        
    def forward(self, f):
        # f: (Batch, factor_dim, T) - TemporalDynamics에서 나온 Clean Factor
        
        h = f
        
        # A. 팩터 정제
        for block in self.blocks:
            h = block(h)
            
        # B. 자산 수익률 복원
        # 여기서 나오는 값은 노이즈가 없는 '설명 가능한 수익률(Systematic Return)'입니다.
        recon_r_o = self.final_conv(h)
        
        return recon_r_o # Shape: (Batch, output_dim, T)