import torch
import torch.nn as nn
import math

class DeepHedgingModelwoPreviousValues(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        '''
        PyTorch에서 모델을 만들 때는 보통 nn.Module이라는 기본 클래스를 상속받습니다. nn.Module 안에는 신경망 모델이 동작하는 데 꼭 필요한 복잡한 기능들(파라미터 추적, GPU 이동, 레이어 등록 등)이 이미 구현되어 있습니다.

        super(...).__init__()을 호출하는 이유는 "내가 만든 모델(DeepHedgingModel)도 nn.Module이 가진 그 기능들을 그대로 초기화해서 쓰겠다"고 선언하는 것입니다.

        2. 코드의 구성 요소
        super(): 부모 클래스(여기서는 nn.Module)를 가리킵니다.

        DeepHedgingModel: 현재 본인이 정의하고 있는 클래스의 이름입니다.

        self: 현재 생성된 인스턴스 자기 자신을 의미합니다.

        .__init__(): 부모 클래스의 초기화 함수(생성자)를 실행하라는 뜻입니다.
        '''

        # LSTM 레이어 정의
        # input_dim: 시장 정보의 수 (예: 주가, 변동성, 이전 델타 등)
        # hidden_dim: 정보를 압축할 은닉 노드 수
        # batch_first=True: 입력 데이터 형태를 (Batch, Time, Feat)로 설정
        # Batch: 샘플 크기 Time: 말 그대로 며칠 동안의 데이터 인지 Feat: 말 그대로 column 개수
        # ex) (1000,20,3): 1000개의 MC 경로가 있고, 20일치이며, 사용하는 데이터가 3개(주가, vol, current delta 처럼)다.
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=3, #hidden layer 개수
                            batch_first=True)

        # 출력 레이어 (델타 산출)
        # 0과 1 사이의 비중을 출력하기 위해 Tanh나 Sigmoid를 쓰기도 하지만,
        # 논문에서는 제약조건에 따라 활성화 함수가 달라질 수 있음
        self.fc = nn.Linear(hidden_dim, output_dim)
        # fc indicates fully connected(dense) layer, 즉 마지막 출력 layer로 바로 앞 hidden layer를 전부 받겠다.

        self.activation = nn.Tanh()
        # 이건 마지막 출력층에만 가해지는 activaion funciton -> 엥 그럼 각 hidden layer는요?
        # 그게 이미 내장되어 있음. 보통 tanh이랑 sigmoid임. -> 엥 왜요?
        # 그게 가장 안정적이고 효과가 좋음. 흔히 수정 안하고, 혹여나 자기가 바꾸고 싶으면 아래와 같이 층을 나눠줘야함.

        '''
        # 이런 식으로 짜면 층 사이에 원하는 함수를 넣을 수 있습니다.
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.relu = nn.ReLU() # 층 사이에 넣고 싶은 함수
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # forward 함수에서
        out, _ = self.lstm1(x)
        out = self.relu(out) # 1층 통과 후 ReLU 적용
        out, _ = self.lstm2(out) # 그 다음 2층 통과
        '''

    def forward(self, x, prev_hedge=None):
        """
        x shape: (Batch Size, Time Steps, Features)
        """
        # 1. LSTM 통과
        # out shape: (Batch, Time, Hidden) -> 모든 시점의 은닉 상태
        # (h_n, c_n): 마지막 시점의 상태 (여기선 사용 안 함)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 2. 델타 계산 (모든 타임 스텝에 대해)
        # Deep Hedging은 만기까지의 '모든 시점'에서 리밸런싱이 일어나므로
        # 전체 시퀀스(lstm_out)를 다 사용합니다.
        deltas = self.activation(self.fc(lstm_out))

        return deltas
    

class DeepHedgingModelWithState(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(DeepHedgingModelWithState, self).__init__()

        # input_dim: 시장 데이터 개수 (예: Log Moneyness + VarSwap Price = 2개)
        self.input_dim = input_dim

        # 실제 입력 차원 계산
        # 시장 데이터(input_dim) + 이전 Stock 델타(1) + 이전 VarSwap N(1) + 잔존만기 tau(1)
        self.total_input_dim = input_dim + 3

        self.hidden_dim = hidden_dim

        # LSTMCell
        self.lstm_cell = nn.LSTMCell(self.total_input_dim, hidden_dim)

        # Normalization
        self.layer_normalization = nn.LayerNorm(hidden_dim)

        # 출력 레이어 (Output dim = 2)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 활성화 함수 (Tanh: -1 ~ 1)
        self.activation = nn.Tanh()


    def forward(self, x, initial_delta=None, initial_N=None, T=None, action_bound=1.0):
        """
        x: (Batch, Steps, input_dim)
        """
        batch_size, steps, _ = x.size()

        # 1. 초기 은닉 상태 초기화
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 2. 초기 포지션 설정
        # prev_delta (Stock): (Batch, 1)
        if initial_delta is None:
            prev_delta = torch.zeros(batch_size, 1, device=x.device)
        else:
            prev_delta = initial_delta

        # prev_N (VarSwap): (Batch, 1)
        if initial_N is None:
            prev_N = torch.zeros(batch_size, 1, device=x.device)
        else:
            prev_N = initial_N

        deltas = []
        var_swap_ns = []

        # dt 계산 (Steps가 N이므로 dt = T/Steps)
        dt = T / steps

        # 3. Time Step Loop
        for t in range(steps):
            # (1) 현재 시점 데이터: (Batch, input_dim)
            market_input = x[:, t, :]

            # shape: (Batch, 1)
            current_time = t * dt
            remaining_time = T - current_time
            tau = torch.full((batch_size, 1), remaining_time, device=x.device)

            # (2) 입력 결합
            # [Market(3), Prev_Stock(1), Prev_Var(1), Tau(1)] -> Total 6
            lstm_input = torch.cat([market_input, prev_delta, prev_N, tau], dim=1)

            # (3) LSTM Cell
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))
            h_t_norm = self.layer_normalization(h_t)

            # (4) Action 출력 및 분리 (Unpacking)
            # fc output: (Batch, 2)
            raw_output = self.fc(h_t_norm)
            action = self.activation(raw_output) * action_bound

            # 텐서 슬라이싱으로 분리 (중요!)
            # action[:, 0]은 (Batch,)가 되므로 unsqueeze(1)을 해줘야 (Batch, 1)이 유지됨
            delta_t = action[:, 0].unsqueeze(1)      # Stock Hedge Ratio
            var_swap_n_t = action[:, 1].unsqueeze(1) # VarSwap Notional

            # (5) 저장
            deltas.append(delta_t)
            var_swap_ns.append(var_swap_n_t)

            # (6) 상태 업데이트 (다음 스텝의 입력으로 사용)
            prev_delta = delta_t
            prev_N = var_swap_n_t

        # 4. 결과 합치기 (Batch, Steps, 1)
        deltas = torch.stack(deltas, dim=1)
        var_swap_ns = torch.stack(var_swap_ns, dim=1)

        return deltas, var_swap_ns
    



#Deep Hedging with Transformer

class DeepHedgingModelTransformerViz(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, output_dim=2, max_len=90):
        super(DeepHedgingModelTransformerViz, self).__init__()

        self.input_dim = input_dim
        self.total_input_dim = input_dim + 3
        self.d_model = d_model

        # 1. Embedding & PE
        self.embedding = nn.Linear(self.total_input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))

        # 2. Transformer Components (수동 분해)
        # nn.TransformerEncoderLayer 대신 직접 정의
        # D1 generating Q/K/V with hyperparameters: d/nhead -> QK' = A -> AV = Z
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(0.1) #일부 뉴런 의도적으로 끄기. 이건 한 뉴런이 학습 전에 꺼질 확률
        self.linear2 = nn.Linear(d_model * 4, d_model)

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation for FFN
        self.relu = nn.ReLU()

        # 3. Output Layer
        self.fc = nn.Linear(d_model, output_dim)
        self.activation = nn.Tanh()

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            # 1. Linear & Embedding 초기화
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # 2. [추가] MultiheadAttention 초기화
            elif isinstance(m, nn.MultiheadAttention):
                # in_proj_weight: Q, K, V를 만드는 가장 중요한 가중치 (3 * d_model, d_model)
                if m.in_proj_weight is not None:
                    nn.init.normal_(m.in_proj_weight, mean=0, std=0.02)

                # bias도 있다면 0으로 초기화
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

                # out_proj는 내부적으로 nn.Linear이므로,
                # 위 loop가 재귀적으로 돌 때 (1)번 조건에서 처리될 수도 있지만,
                # 명시적으로 확실히 하고 싶다면 여기서 해도 됩니다.
                # (보통 PyTorch의 재귀적 modules() 호출 덕분에 out_proj는 (1)에서 처리됩니다.)

            # 3. Positional Encoding 파라미터
            elif isinstance(m, nn.Parameter):
                pass

        # Positional Encoding 명시적 초기화
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

    def forward(self, x, initial_delta=None, initial_N=None, T=None, action_bound=1.0):
        batch_size, steps, _ = x.size()
        device = x.device

        # 초기값 설정 (생략 가능, 위 코드와 동일)
        prev_delta = initial_delta if initial_delta is not None else torch.zeros(batch_size, 1, device=device)
        prev_N = initial_N if initial_N is not None else torch.zeros(batch_size, 1, device=device)

        deltas = []
        var_swap_ns = []

        # *** Attention Map을 저장할 리스트 ***
        # 매 Time Step마다 모델이 어디를 쳐다봤는지 기록
        causal_attention_map = torch.zeros(steps, steps)

        final_inputs = torch.empty(batch_size, 0, self.d_model, device=device)
        dt = T / steps

        for t in range(steps):
            # (1) Input 구성 & Embedding
            market_input = x[:, t, :]
            current_time = t * dt
            remaining_time = T - current_time
            tau = torch.full((batch_size, 1), remaining_time, device=device)

            current_raw_input = torch.cat([market_input, prev_delta, prev_N, tau], dim=1)
            current_embed = self.embedding(current_raw_input).unsqueeze(1) * math.sqrt(self.d_model)
            current_embed = current_embed + self.pos_encoder[:, t:t+1, :]

            final_inputs = torch.cat([final_inputs, current_embed], dim=1)

            # (2) Multi-Head Attention (수동 호출)
            # need_weights=True로 설정해야 가중치(Map)가 나옴
            # attn_output: (Batch, Seq_Len, d_model)
            # attn_weights: (Batch, Seq_Len, Seq_Len) -> 이게 바로 Attention Map!
            attn_output, attn_weights = self.mha(final_inputs, final_inputs, final_inputs, need_weights=True)
            
            # Residual Connection + LayerNorm
            # final_inputs로 미분될 때 gradient vanishing 방지를 위해
            # skip connection
            x_sub = self.norm1(final_inputs + attn_output)

            # (3) Feed Forward
            # noise(wx+b = 음수, 필요없음)날리고, 미분값(if 양수) = 1로 해서 빠르게 학습 시키기 위해
            ff_output = self.linear2(self.dropout(self.relu(self.linear1(x_sub))))
            transformer_output = self.norm2(x_sub + ff_output)

            # (4) Attention Map 저장 (가장 마지막 시점 t가 과거를 어떻게 보는지)
            # attn_weights[:, -1, :] : 현재 시점(마지막 행)이 과거 모든 시점에 준 가중치
            causal_attention_map[t, :t+1] = attn_weights[0, -1, :].detach().cpu()

            # (5) Action 도출
            last_hidden_state = transformer_output[:, -1, :]
            raw_output = self.fc(last_hidden_state)
            action = self.activation(raw_output) * action_bound

            delta_t = action[:, 0].unsqueeze(1)
            var_swap_n_t = action[:, 1].unsqueeze(1)

            deltas.append(delta_t) #At each t node, each deltas of samples
            var_swap_ns.append(var_swap_n_t)

            prev_delta = delta_t
            prev_N = var_swap_n_t

        deltas = torch.stack(deltas, dim=1)
        var_swap_ns = torch.stack(var_swap_ns, dim=1)

        # 모델이 뱉는 값에 attention_maps 추가
        return deltas, var_swap_ns, causal_attention_map

"""
class DeepHedgingModelTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, output_dim=2, max_len=90): #d_model = hidden_dim
        super(DeepHedgingModelTransformer, self).__init__()

        # 1. 입력 차원 정의
        self.input_dim = input_dim
        # 실제 입력: Market(input_dim) + Prev_Stock(1) + Prev_Var(1) + Tau(1) = input_dim + 3
        self.total_input_dim = input_dim + 3

        self.d_model = d_model

        # 2. Embedding Layer
        # Raw Input을 Transformer 내부 차원(d_model)으로 뻥튀기 해주는 역할
        # (LSTM은 그냥 넣어도 되지만 Transformer는 차원을 맞춰주는 게 성능상 유리함)
        self.embedding = nn.Linear(self.total_input_dim, d_model)

        # 3. Positional Encoding
        # 순서 정보를 주입 (학습 가능한 파라미터로 설정)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model)) # max_len x d_model 0 mtx 1개

        # 4. Transformer Encoder
        # batch_first=True: (Batch, Steps, Features) 형태 유지
        # EncoderLayer는 후술할 D1 D2 D3를 DEF / 그냥 Encoder는 그런 Encoderlayer를 몇겹 쌓을지 결정)

        # initial W를 통과하고 t by d(=d_model)이 된 mtx에다가, D1 = d x d / D2 = d x dim_feedforward(관습적으로 d*4) / D3 = d*4 x d mtx에 차례대로 projection
        # D1: 'Multi Head Attention', 말그대로 본격적으로 d로 뻥튀기된 데이터 정보들에 대해 '어떻게 바라볼까?' 하는 거.
        # 이 때 nhead가 들어오는데 nhead = 'data를 몇개의 group(perspective)에서 분석할까?'를 정하는 거고, 각 그룹이 d/n = k차원을 갖는 것 현재는 n=8개의 관점에서, 각 관점마다 8(64/8)개의 차원에서 분석하자
        # 무조건 d/n % = 0이어야 하고, 보통 8~64 dim per 1 head가 golden rule.
        # D2는 말그대로 고차원으로 걍 한번 projection 해주는 역할

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 

        # 5. 출력 레이어
        self.fc = nn.Linear(d_model, output_dim)

        # 6. 활성화 함수
        self.activation = nn.Tanh()

        # 가중치 초기화 (선택 사항)
        self._init_weights()

    def _init_weights(self):
      for m in self.modules():
          # Linear 레이어나 Embedding 레이어인 경우
          if isinstance(m, (nn.Linear, nn.Embedding)):
              nn.init.normal_(m.weight, mean=0, std=0.02)
              if m.bias is not None:
                  nn.init.zeros_(m.bias) # 편향은 보통 0으로 시작
          # Positional Encoding 파라미터인 경우
          elif isinstance(m, nn.Parameter):
              # 이 부분은 수동으로 지정한 경우라 따로 체크 필요
              pass

      # PE는 따로 한 번 더 확실히 해줌, 얘는 nn.parameter로 부른거라,module로 안잡힘 즉 layer가 아니라 안잡힘.
      nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

    def generate_square_subsequent_mask(self, sz_times, device='cpu'): #지금 당장은 어쩌피 loop 돌면서 attention map 마지막 행만 쓰므로 필요가 없음.
        # 미래 정보를 못 보게 가리는 마스크 (Upper Triangular Mask)
        # 대각선 위쪽을 -inf로 채움
        mask = (torch.triu(torch.ones(sz_times, sz_times, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, initial_delta=None, initial_N=None, T=None, action_bound=1.0):

        # x: (Batch, Steps, input_dim) -> Market Data는 이미 다 알고 있음
        # 하지만 Action은 Loop를 돌면서 하나씩 채워나가야 함.

        # batch_size, steps, _ = x.size()
        # device = x.device

        # 1. 초기 포지션 설정
        if initial_delta is None:
            prev_delta = torch.zeros(batch_size, 1, device=device)
        else:
            prev_delta = initial_delta

        if initial_N is None:
            prev_N = torch.zeros(batch_size, 1, device=device)
        else:
            prev_N = initial_N

        # 결과를 저장할 리스트
        deltas = []
        var_swap_ns = []

        # Transformer에 넣을 History를 저장하는 버퍼
        # (Batch, 0, total_input_dim)에서 시작해서 매 스텝 쌓아감
        final_inputs = torch.empty(batch_size, 0, self.d_model, device=device)

        dt = T / steps

        # --- Time Step Loop (Autoregressive) ---
        for t in range(steps):
            # (1) 현재 시점의 입력 Feature 구성
            market_input = x[:, t, :] # (Batch, input_dim)

            current_time = t * dt
            remaining_time = T - current_time
            tau = torch.full((batch_size, 1), remaining_time, device=device)

            # [Market, Prev_Stock, Prev_Var, Tau] 결합 -> (Batch, Total_Input_Dim)
            current_raw_input = torch.cat([market_input, prev_delta, prev_N, tau], dim=1)

            # (2) Embedding & Scaling
            # (Batch, 1, d_model)
            current_embed = self.embedding(current_raw_input).unsqueeze(1) * math.sqrt(self.d_model)
            # root(d)를 곱해주는 건, embed가 pe에 지지 않게 하기 위해.
            # 기존의 pe방식인 sin/cos는 -1~1의 값이 나와 초기값이 임베딩 값보다 커질 가능성 O
            # 따라서 이걸 곱해서 임베드도 SCALING해줌. 우리야 0언저리에서 시작하는 learnable PE로 해서 상관 없지만, 있어도 무관하니 남

            # (3) Positional Encoding 더하기
            # 현재 시점 t에 해당하는 pos_encoding만 가져와서 더함
            current_embed = current_embed + self.pos_encoder[:, t:t+1, :]

            # (4) History Buffer에 추가
            # 이제 history_buffer는 t=0부터 t까지의 정보를 담고 있음 shape: (Batch, t+1, d_model)
            final_inputs = torch.cat([final_inputs, current_embed], dim=1)

            # (5) Transformer Encoder 통과
            # 매 스텝마다 전체 History를 다 다시 봄 (Self-Attention)
            # 마스크는 사실 Loop 방식이라 없어도 되지만, 습관적으로 넣거나 패딩용으로 쓸 수 있음.
            # 여기서는 causal 구조상 Loop를 돌리므로 t시점 output만 잘 뽑으면 됨.

            # (Batch, t+1, d_model)
            transformer_output = self.transformer_encoder(final_inputs)

            # (6) 현재 시점(t)의 결과만 가져오기
            # 마지막 시점의 Hidden State 사용
            last_hidden_state = transformer_output[:, -1, :] # (Batch, d_model)

            # (7) Action 도출
            raw_output = self.fc(last_hidden_state)
            action = self.activation(raw_output) * action_bound

            # (8) Unpacking & Save
            delta_t = action[:, 0].unsqueeze(1)
            var_swap_n_t = action[:, 1].unsqueeze(1)

            deltas.append(delta_t)
            var_swap_ns.append(var_swap_n_t)

            # (9) 상태 업데이트
            prev_delta = delta_t
            prev_N = var_swap_n_t

        # 결과 합치기
        deltas = torch.stack(deltas, dim=1)
        var_swap_ns = torch.stack(var_swap_ns, dim=1)

        return deltas, var_swap_ns #2dim
"""