import torch

def PCA(data_tensor, k):
    """
    [Standard Approach]
    논문(Bai & Ng 2002 등)에서 사용하는 표준적인 방식으로 잔차 분산(sigma^2)을 계산합니다.
    
    Args:
        data_tensor: (Batch, d, T) - 학습 데이터 전체
        k: 팩터 개수 (num_factors)
        
    Returns:

        cov_full: (d, d) - PCA Result

        eig_vals_arranged: (n, n) - eigenvalues mtx where n is the number of eigenvalues and [diag]i >= [diag]j whose i > j
        
        diag_variances: (d, ) - Resid variances of each stock
    """
    print("Start to conduct the PCA. Please check the data's shape is correctly setted for your purpose." )

    # 1. 데이터 차원 변환: (Batch, d, T) -> (d, Total Samples)
    # 공분산을 구하기 위해 모든 배치와 시간축을 '샘플'로 폅니다.
    B, d, T = data_tensor.shape
    X = data_tensor.permute(1, 0, 2).reshape(d, -1)
    
    # 2. 전체 공분산 행렬 계산 (Sample Covariance)
    # shape: (d, d)
    cov_full = torch.cov(X)
    
    # 3. 고유값 분해 (Eigendecomposition)
    # eig_vals는 오름차순으로 정렬되어 나옵니다 (작은거 -> 큰거)
    eig_vals, eig_vecs = torch.linalg.eigh(cov_full)

    eig_vals_arranged = eig_vals.flip(dims=(0,))
    
    # 4. 팩터 부분 복원 (Reconstruct Factor Covariance)
    # 가장 큰 k개의 고유값과 고유벡터만 사용
    # U_k: (d, k), L_k: (k,)
    U_k = eig_vecs[:, -k:]
    L_k = torch.diag(eig_vals[-k:])
    
    # Sigma_factor = U_k * Lambda_k * U_k.T
    cov_factor = U_k @ L_k @ U_k.T
    
    # 5. 잔차 공분산 계산 (Residual Covariance)
    # Sigma_residual = Sigma_total - Sigma_factor
    cov_residual = cov_full - cov_factor
    
    # 6. 대각 성분(Variance)만 추출 결과값 1차원 벡터
    # 나중에 diag mtx화 하고 싶으면 그냥 다시 1차원에 torch.diag(x) 하면됨.
    # 비대각 성분(Covariance)은 과감하게 버립니다 (Assumption: Diagonal Matrix)
    diag_variances = cov_residual.diag()
    
    # 7. 수치적 안정성 (Numerical Stability)
    # 계산 오차로 인해 아주 작은 음수가 나올 수 있으므로 0보다 큰 최소값으로 클리핑
    diag_variances = diag_variances.clamp(min=1e-6)

    print(f"[PCA Finish] Used top-k={k} factors.")
    print(f"Top 5 Eigenvalues: {eig_vals_arranged[:5].tolist()}") # 상위 5개 미리보기
    print(f'The returns : cov_full / eig_vals_arranged / diag_variances')

    return cov_full, eig_vals_arranged, diag_variances