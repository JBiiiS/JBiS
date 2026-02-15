import torch
import torch.nn as nn
import math

'''

기본적으로 GaussainFourierProjection으로 많은 sin cos 파동 생성. 물론 동일한 샘플에는 동일한 t가 적용되므로 어떤 t를 가정하였을 때, 해당 t에 n(>>k)개의 가중치를 곱함.(예컨대, 현재 코드는 정규분포로 생성. 이후 * 30을 해서 전반적인 진동폭을 키워줌) 이후에 *2pi를 한다음 sin cos을 가하면 가중치만큼의 주기가 나옴. 이후에 n by k 곱해서 각 샘플 feature 차원만큼 가해줌. 

"왜 하나의 샘플에 적용되는 t(noise 횟수)는 동일한데 각 행에(d by t 기준) 전부 같은 가중치를 주지 않나요?": 그렇게 하면 위치정보로써 전달할 수 있는 차원이 딱 하나라서 학습이 잘 안될뿐더러, 모든 feature가 동일한 scale만큼 커지고 줄어들면 이게 reigme이 바뀌거나 전반적인 factor의 영향으로 오인할 가능성 o

"그럼 한 샘플 안에서 각 feature마다 다른 위치정보 수치를 주는 이유가 무엇인가요?" 위와 거의 마찬가지. 예컨대 동일한 방향으로 키워버리면, '어 이거 둘다 늘어난 거 보니 지금 그냥 주식이 그런가보네?' 하는데, 왔다갔다하면 '아 대부분은 위치정보고 이 중간 어딘가에 노이즈겠네'로 학습가능.

범용성을 위해 분리해서 제작하였으며, 만약에 배치마다 다르다? 하면 배치만큼 주고, 나중에 나온 마지막 return에 대하여 unsqueeze(2) : (batch, k, 1)하면 되는거고, 만약에 샘플마다 똑같은 거 주는데, 샘플 안에서 다르다 하면 (ex) transformer), 그냥 똑같이 (1,..,30) 주고 factor_dim = 1로해서 만든다음, unsqueeze(0) 하면됨. 

'''

# ---------------------------------------------------------
# 1. 기초 부품: 시간 임베딩 생성기 (Learnable)
# ---------------------------------------------------------
class GaussianFourierProjection(nn.Module):
    """
    t (Scalar) -> Vector 변환
    학습 불가능한 랜덤 가중치로 t를 고차원으로 뻥튀기해줍니다. (Transformer의 Positional Encoding 같은 역할)
    """
    def __init__(self, time_emb_dim, scale=10.):
        super().__init__()
        # 학습되지 않는(requires_grad=False) 랜덤 가중치 고정
        self.W = nn.Parameter(torch.randn(time_emb_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        # x: (Batch, ): t들이 하나씩 담겨있음
        # x_proj: (Batch, time_embed_dim // 2)
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi #한 행에서 똑같은 t에 서로다른 sin/cos 값 생성
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ---------------------------------------------------------
# 2. 기초 부품: 1D ResNet 블록
# ---------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, factor_dim, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(factor_dim, factor_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(factor_dim, factor_dim, 3, padding=1)
        self.act = nn.GELU()

        '''
        "What is Conv1d layer?"

        기본적인 structure는 nn.Conv1d(in_channel_dim, out_channel_dim, kernel_size , padding)임. 우선 padding은 input에 대해 좌우에 그 숫자만큼 가상의 0을 붙여준다고 보면 되고, kernel size는 가중치 mtx의 개수이자, x에 대한 열쪽 window임. 예컨대 지금 3,1은, 어떤 시점 data 기준으로 그 앞과 뒤까지 같이 보겠다. 이렇게 되는거고, 출력되는 mtx의 열계수는 (input열계수 + 2* padding -(커널사이즈 -1))임. padding이 0이면 지금 대로면 t-2개가 된다는 걸 알텐데, 그래서 padding =1 을 해줘서 맨 처음과 마지막엔 가상의 0이 있다고 보게 하는 것.

        이을러서 weight mtx 는 (out_channel_dim, in_channel_dim)의 크기임. 즉 반드시 in_channel_dim = input의 raw rank여야함. out_channel_dim이 output의 raw rank가 되는데, 즉, 주어진 input mtx의 행계수를 변형시키는 역할을 가지고 있다고 보면됨.
        '''
        
        # 시간 정보가 들어오면 채널 수에 맞게 변환해주는 층
        self.time_mlp = nn.Linear(time_emb_dim, factor_dim)

    def forward(self, x, t_emb):
        # x: (Batch, k, T)
        h = self.conv1(self.act(x))
        
        # [Broadcasting] (Batch, time_dim) -> (Batch, k, 1) 로 늘려서 더함
        h += self.time_mlp(t_emb)[:, :, None]
        
        h = self.conv2(self.act(h))
        return x + h  # Residual Connection
    
    # 결국, conv1  / conv2가 노이즈 걷어내기 + 위치 정보 해석하기의 핵심적 역할을 수행.

# ---------------------------------------------------------
# 3. [NEW] 핵심 모듈: 시간 역학 처리기 (Time-Aware Dynamics)
# ---------------------------------------------------------
class TemporalDynamics(nn.Module):
    """
    [User's Request]
    Encoder 뒤에 붙어서, 시간 정보(t)와 팩터(x)를 받아
    시계열적인 흐름(Dynamics)을 학습하고 주입해주는 독립적인 클래스입니다.
    다른 모델에서도 '시간 정보 섞기' 용도로 재사용 가능합니다.
    """
    def __init__(self, factor_dim, time_emb_dim, num_layers=4):
        super().__init__()
        
        # 1. 시간 임베딩 처리기 (MLP)
        # Raw t -> Sin/Cos -> MLP -> t_emb
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(time_emb_dim), 
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 2. Dynamics 학습 층 (Conv1d Layers)
        self.layers = nn.ModuleList([
            TemporalBlock(factor_dim, time_emb_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, t):
        """
        Args:
            x: (Batch, k, T) - Encoder를 통과한 Latent Factor
            t: (Batch, )     - 노이즈 단계 (0 ~ 999)
        Returns:
            out: (Batch, k, T) - 시간 정보가 주입되고 흐름이 학습된 Factor
        """
        # A. t를 벡터로 변환: (Batch,) -> (Batch, time_emb_dim)
        t_emb = self.time_mlp(t)
        
        # B. 레이어 통과 (x의 크기에 맞춰서 t_emb가 알아서 Broadcasting됨)
        f_0_hat = x
        for layer in self.layers:
            f_0_hat = layer(f_0_hat, t_emb)
            
        return f_0_hat