import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import FinanceDataReader as fdr  # type: ignore
import random
from tqdm import tqdm

def clean_price_data(price_series):
    # 1. Pandas Series로 변환 (아직 아니라면)
    if not isinstance(price_series, (pd.DataFrame,pd.Series)):
        try:
            s = pd.Series(price_series)
        
        
        except:
        
            try:
                s = pd.DataFrame(price_series)  
        
            except Exception as e:
                raise ValueError(f"cannot be pd: {e} // {type(e)}")
            
    
    s = price_series
    

    # 2. 0 또는 음수값이 있다면 NaN으로 처리 (비정상 데이터 제거)
    s[s <= 0] = np.nan
    print(f'The number of Nulls : {s.isnull().sum()}')

    # 3. 선형 보간 (Linear Interpolation)
    # 주변 가격의 평균으로 메꿈으로써 수익률의 급격한 왜곡 방지
    s = s.interpolate(method='linear')

    # 4. 양 끝단에 남은 결측치 처리 (보통 bfill/ffill로 마무리)
    s = s.bfill().ffill()

    
    # 5. data 개수: 3000개 이하
    # print(f'The number of data points : {len(s)}')
    # if len(s) > 3000:
    #     s = s.tail(3000)

    return s

def fdr_data_wo_ticker(TARGET_N_STOCKS, START_DATE, END_DATE, MKT):

    # ---------------------------------------------------------
    # 1. MKT 종목 리스트 가져오기
    # ---------------------------------------------------------
    print(f"{MKT} 종목 리스트를 불러오는 중...")
    try:
        stock_list = fdr.StockListing(MKT)
        all_tickers = stock_list['Symbol'].tolist()
        print(f"총 {len(all_tickers)}개의 종목을 찾았습니다.")
    except Exception as e:
        print(f"{MKT} 리스트 로드 실패 (인터넷 연결 확인 필요): {e}")
        # 테스트용 더미 리스트 (실패 시 비상용)
        all_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'LLY', 'V']

    # ---------------------------------------------------------
    # 2. 랜덤하게 n개 뽑기
    # ---------------------------------------------------------
    if TARGET_N_STOCKS > len(all_tickers):
        return print(f'The targeted number of stocks is more than all stocks in {MKT} \n Target: {TARGET_N_STOCKS} | All: {len(all_tickers)}')
    else:
        selected_tickers = random.sample(all_tickers, ((TARGET_N_STOCKS * 1.3)//1))
    

    print(f"다운로드 시도할 종목 수: {len(selected_tickers)}")
    print(f"예시: {selected_tickers[:5]} ...")

    # ---------------------------------------------------------
    # 3. 데이터 다운로드 및 병합
    # ---------------------------------------------------------
    close_data_dict = {}

    print("데이터 다운로드 시작...")
    for ticker in tqdm(selected_tickers):
        try:
            df = fdr.DataReader(ticker, START_DATE, END_DATE)
            
            # 데이터가 없으면 배제
            if not df.empty and 'Close' in df.columns:
                close_data_dict[ticker] = df['Close']
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

    # ---------------------------------------------------------
    # 6. DataFrame 생성 및 전처리
    # ---------------------------------------------------------
    price_df = pd.DataFrame(close_data_dict)

    null_threshold = 10
    null_counts = pd.Series(price_df.isnull().sum())

    # Null이 기준치 이하인 우량 종목들만 남김
    valid_tickers = null_counts[null_counts <= null_threshold].index
    dropped_tickers = list(set(price_df.columns) - set(valid_tickers))

    price_df = price_df[valid_tickers]

    print(f" -> Null {null_threshold}개 초과 종목 {len(dropped_tickers)}개 제거됨")
    if len(dropped_tickers) > 0:
        print(f"    (제거된 종목 예시: {dropped_tickers[:5]} ...)")

    return price_df



def fdr_data_with_ticker(START_DATE, END_DATE, TICKER):

    # ---------------------------------------------------------
    # 1. MKT 종목 리스트 가져오기
    if type(TICKER) == list:
        close_data_dict = {}
        for ticker in TICKER:
            df = fdr.DataReader(ticker, START_DATE, END_DATE)
            close_data_dict[ticker] = df['Close']


        price_df = pd.DataFrame(close_data_dict)

    elif type(TICKER) == str:
        price_df = pd.Series((fdr.DataReader(TICKER, START_DATE, END_DATE))['Close'])

    # ---------------------------------------------------------
    # 2. 전처리
    # ---------------------------------------------------------
    if type(TICKER) == list:
        null_threshold = 10
        null_counts = pd.Series(price_df.isnull().sum())

        # Null이 기준치 이하인 우량 종목들만 남김
        valid_tickers = null_counts[null_counts <= null_threshold].index
        dropped_tickers = list(set(price_df.columns) - set(valid_tickers))

        price_df = price_df[valid_tickers]

        print(f" -> Null {null_threshold}개 초과 종목 {len(dropped_tickers)}개 제거됨")
        if len(dropped_tickers) > 0:
            print(f"    (제거된 종목 예시: {dropped_tickers[:5]} ...)")

    return price_df



def rolling_rv(prices, window=10):
    if isinstance(prices, (np.ndarray)):
        prices = torch.from_numpy(prices)
    
    assert isinstance(prices, (torch.tensor, )), f"The input's type is not tensor (type: {type(prices)})." 


    # prices shape: (Batch, Total_Steps) 이라고 가정
    # 1. Log Return 계산
    # prices가 float64일 수 있으니 안전하게 .float() 처리
    prices = prices.float()
    ln_prices = torch.log(prices)
    ln_diff = ln_prices[:, 1:] - ln_prices[:, :-1] # (Batch, Total_Steps-1)

    # 2. Unfold를 사용하여 Rolling Window 만들기
    # dimension 1(시간축)에서 window 사이즈만큼 묶음
    # step=1: 한 칸씩 이동
    windows = ln_diff.unfold(dimension=1, size=window, step=1)
    # windows shape: (Batch, Number_of_Windows, window)

    # 3. 각 윈도우의 표준편차(std) 혹은 분산(var) 계산
    # Deep Hedging에서는 보통 Std를 많이 쓰지만, RV 개념상 분산이 필요하면 var 유지
    # 여기서는 질문주신 코드대로 var 유지하되, 차원 유지
    rolling_vars = torch.std(windows, dim=2) # (Batch, Number_of_Windows)

    # 4. 차원 확장 (Batch, Steps, 1) - Concat을 위해
    rolling_vars = rolling_vars.unsqueeze(-1)

    return rolling_vars


def log_data(data):
    """
    주가 데이터를 받아 로그 가격과 로그 차분(수익률)을 반환합니다.
    
    Args:
        data (pd.Series or pd.DataFrame): 주가 데이터
        
    Returns:
        ln_price (pd.Series or pd.DataFrame): 로그 가격
        ln_price_diff (pd.Series or pd.DataFrame): 로그 차분 (첫 행은 NaN)
    """
    # 1. 로그 가격 계산: ln(P)
    ln_price = np.log(data)
    
    # 2. 로그 차분 계산: ln(P_t) - ln(P_{t-1})
    # pandas의 diff() 함수를 사용하면 간편합니다.
    ln_price_diff = ln_price.diff()
    
    return ln_price, ln_price_diff

class To_TensorSet:
    def __init__(self, config):
        self.t = config.steps       
        self.d = config.num_assets  
        
    def process(self, data, init_shuffle = False):
        # 1. 데이터 타입 체크 및 (Total_Time, D) 형태로 통일
        # -------------------------------------------------------------------------
        # [수정] torch.Tensor 입력 처리 추가
        # -------------------------------------------------------------------------
        print(f'Please check the input shape : \n the current type: {type(data)} | the current shape: {data.shape} \n you must put (t,) / (t,d) / (b,t,d)')

        # 1. 데이터 타입 체크 및 (Total_Time, D) 형태로 통일

        if isinstance(data, torch.Tensor):
            # GPU에 있을 수 있으므로 cpu로 이동 후 numpy 변환
            # (B, T, D)인 경우 -> (B*T, D)로 펴서 2차원으로 만듦 (뒤쪽 로직 호환성 유지)

            if data.dim() == 3: 
                raw_data = data.reshape(-1, data.size(-1)).detach().cpu().numpy()
            elif data.dim() == 2: # (T, D)
                raw_data = data.detach().cpu().numpy()
            elif data.dim() == 1: # (T,)
                raw_data = data.detach().cpu().numpy().reshape(-1, 1)
        

        elif isinstance(data, pd.Series): # (t,)
            raw_data = data.values.reshape(-1, 1)
            
        elif isinstance(data, pd.DataFrame): # (t,d)
            raw_data = data.values
            
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                raw_data = data.reshape(-1, 1)
            elif data.ndim == 3: 
                # 혹시 numpy도 (B, T, D)로 들어올 경우를 대비해 2차원으로 펼침
                raw_data = data.reshape(-1, data.shape[-1])
            else:
                raw_data = data
        else:
            raise ValueError("Please check the input.")

        # Asset 개수 검증
        L, current_d = raw_data.shape
        if current_d != self.d:
            print(f"Warning: Input data dim ({current_d}) does not match config.num_assets ({self.d}). Adjusting to data dim.")
            self.d = current_d

        # 2. Window Slicing (t씩 자르기)
        num_samples = L // self.t
        if num_samples < 2:
            raise ValueError(f"Not enough data. Total len: {L}, Step: {self.t}")

        cutoff = num_samples * self.t
        trimmed_data = raw_data[:cutoff] 
        # (N, T, D) 형태의 텐서 생성
        tensor_data = torch.tensor(trimmed_data, dtype=torch.float32).view(num_samples, self.t, self.d)

        # 3. Train / Test Split
        # 마지막 1개 윈도우는 Test, 나머지는 Train
        test_set_btd = tensor_data[-1:]   # (1, T, D)
        train_set_btd = tensor_data[:-1]  # (N-1, T, D)
    
        # 4. Shuffle (Train set only, if wanted)
        if init_shuffle == True:
            indices = torch.randperm(train_set_btd.size(0))
            train_set_btd = train_set_btd[indices]
        
        train_set_bdt = train_set_btd.permute(0, 2, 1) # (N, D, T)
        test_set_bdt = test_set_btd.permute(0, 2, 1) # (1, D, T)

        return {
            'train': {
                'BTD': train_set_btd, # List of Tensors (가변 배치)
                'BDT': train_set_bdt, # List of Tensors (가변 배치) 
            },
            'test': {
                'BTD': test_set_btd,
                'BDT': test_set_bdt
            }
        }