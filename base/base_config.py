import torch
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass 
class BaseConfig:
    # ----------------------------------
    # [1] 공통 환경 설정 (Environment)
    # ----------------------------------
    project_name: str
    experiment_name: str 

    seed: int = 42
    
    # device는 저장(Serialization)을 위해 str로 관리하고,
    # 실제 사용할 때 torch.device로 변환하는 프로퍼티를 씁니다.
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------
    # [2] 데이터 공통 설정 (Data)
    # ----------------------------------
    num_assets: int = 100       # d: 자산 개수
    batch_size: int = 16
    total_samples: int = 20000
    T: float = 30/252           # T: 윈도우 길이   
    steps: int = 30           # N: steps
    rv_window: int = 10
    r: float = 0.0
    num_epochs: int = 1000
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3   # Optimizer 설정
    dropout: float = 0.1

    
    # ----------------------------------
    # [3] 유틸리티 메서드 (필수!)
    # ----------------------------------
    @property # making a function call look like a variable access
    def device(self):
        """실제 코드에서 cfg.device로 접근할 때 호출됨"""
        return torch.device(self.device_name)

    def to_dict(self):
        """설정값을 딕셔너리로 변환 (로그 저장용)"""
        return asdict(self)
    
    def __post_init__(self):
        # 객체가 생성된 직후에 자동으로 실행됨
        # 여기서 의존성 있는 계산을 수행합니다.
        self.dt = self.T / self.steps