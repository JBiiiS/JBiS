from dataclasses import dataclass, asdict
from base.base_config import BaseConfig
from typing import Optional


@dataclass
class DiffusionConfig(BaseConfig):
    # ----------------------------------
    # [1] 모델 구조 (Architecture)
    # ----------------------------------
    model_type: str = "ScoreNet"
    num_factors: int = 16       # k: 팩터 개수 (Bottleneck)
    hidden_dim: int = 128       # U-Net 내부 채널 수

    # ----------------------------------
    # [2] 확산 프로세스 (Diffusion SDE)
    # ----------------------------------
    sde_steps: int = 1000       # N: SDE 스텝 수(0 ~ 1 devision)
    eta_min: float = 0.1       # VP-SDE 최소 노이즈
    eta_max: float = 20.0      # VP-SDE 최대 노이즈 
    # eta function standard: eta_min + (eta_max -eta_min)*t
    
    
    # PCA 초기화 관련 (sigma^2 계산용)
    precompute_sigma: bool = True

    def __post_init__(self):
        super().__post_init__() # 부모(BaseConfig)의 dt 계산 실행
        # SDE용 delta_t 계산
        self.sde_dt = 1.0 / self.sde_steps
    
    def to_dict(self):
        """설정값을 딕셔너리로 변환 (로그 저장용)"""
        return asdict(self)