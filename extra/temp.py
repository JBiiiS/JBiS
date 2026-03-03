#type:ignore

# ─────────────────────────────────────────
# 5. 학습 루프 (Algorithm 3)
# ─────────────────────────────────────────
def train_dlqf(
    model: DLQFRNN,
    X: torch.Tensor,    # (N, T, input_dim)
    Y: torch.Tensor,    # (N, M) — 정렬된 실제 |r| × scale_factor
    config: DLQFRNNConfig,
    device: str
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    M = config.total_quantile
    alpha = torch.linspace(1 / M, 1, M).to(device)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    # Algorithm 3: 초기화 안정성 체크
    print("초기화 조건 확인 중...")
    for attempt in range(10):
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        model.eval()
        with torch.no_grad():
            x_s = X[:1].to(device)
            y_s = Y[:1].to(device)
            r_q = model(x_s, alpha)
            loss_val = l2_distance_loss(y_s, r_q).item()
            std_val = r_q.std().item()

        print(f"  시도 {attempt+1}: loss={loss_val:.4f}, std={std_val:.6f}")
        if loss_val <= 1e2 and std_val >= 1e-3:
            print("  ✅ 초기화 조건 통과\n")
            break

    # 본격 학습
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            r_q = model(x_batch, alpha)
            loss = l2_distance_loss(y_batch, r_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] | L2 Loss: {epoch_loss/len(loader):.6f}")

    return model


# ─────────────────────────────────────────
# 6. Sanity Check
# ─────────────────────────────────────────
if __name__ == "__main__":
    cfg = DLQFRNNConfig()

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")




arameters: 7,111
Epoch [  5/50] | Train 0.085958 | Val 0.099844 | Best 0.099844
Epoch [ 10/50] | Train 0.077798 | Val 0.088019 | Best 0.084881
Epoch [ 15/50] | Train 0.069890 | Val 0.094227 | Best 0.084881

Early stopping at epoch 17

=============================================
Test L2 Loss : 0.050050
Test MSE     : 0.00096530
Test MAE     : 0.00008284
Test MAPE    : 1.0215
=============================================

Gamma Statistics:
Mean Gamma   : 0.205121
Std Gamma    : 0.029250
Min Gamma    : 0.165580
Max Gamma    : 0.312709



cfg.learning_rate = 1e-2
cfg.seed = 42
cfg.num_epochs = 50
device = cfg.device
cfg.num_layers_lstm = 3
cfg.hidden_dim = 10
cfg.latent_dim_1 = 16
cfg.latent_dim_2 = 8
cfg.latent_dim_3 = 4
cfg.exp_deno_init = 8.5
cfg.use_learnable_exp = True
cfg.use_non_learnable_exp = False
PATIENCE = 10
no_improve = 0


Parameters: 7,111
Epoch [  5/50] | Train 0.076413 | Val 0.109412 | Best 0.086635
Epoch [ 10/50] | Train 0.064977 | Val 0.077916 | Best 0.077916
Epoch [ 15/50] | Train 0.066465 | Val 0.100138 | Best 0.073230
Epoch [ 20/50] | Train 0.061563 | Val 0.095045 | Best 0.069853
Epoch [ 25/50] | Train 0.057913 | Val 0.064205 | Best 0.064205
Epoch [ 30/50] | Train 0.056800 | Val 0.071996 | Best 0.064205
Epoch [ 35/50] | Train 0.057467 | Val 0.077131 | Best 0.064205

Early stopping at epoch 35

=============================================
Test L2 Loss : 0.047529
Test MSE     : 0.00077577
Test MAE     : 0.00007457
Test MAPE    : 0.6638
=============================================

Gamma Statistics:
Mean Gamma   : 0.133151
Std Gamma    : 0.083014
Min Gamma    : 0.013576
Max Gamma    : 0.481118






cfg.num_layers_lstm = 3
cfg.hidden_dim = 10
cfg.latent_dim_1 = 16
cfg.latent_dim_2 = 8
cfg.latent_dim_3 = 4
cfg.exp_deno_init = 8.78
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
PATIENCE = 10
no_improve = 0

Parameters: 7,111
Epoch [  5/50] | Train 0.076308 | Val 0.105506 | Best 0.086227
Epoch [ 10/50] | Train 0.066939 | Val 0.089294 | Best 0.084200
Epoch [ 15/50] | Train 0.067758 | Val 0.072435 | Best 0.071604
Epoch [ 20/50] | Train 0.059476 | Val 0.081874 | Best 0.064091
Epoch [ 25/50] | Train 0.056367 | Val 0.067593 | Best 0.064091


Early stopping at epoch 26

=============================================
Test L2 Loss : 0.048737
Test MSE     : 0.00089889
Test MAE     : 0.00007841
Test MAPE    : 0.6434
=============================================

Gamma Statistics:
Mean Gamma   : 0.143896
Std Gamma    : 0.086289
Min Gamma    : 0.004539
Max Gamma    : 0.375434












-----------------------------------------------------




cfg.num_layers_lstm = 5
cfg.hidden_dim = 10
cfg.latent_dim_1 = 16
cfg.latent_dim_2 = 8
cfg.latent_dim_3 = 4
cfg.exp_deno_init = 8.5
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = False
PATIENCE = 20
no_improve = 0



Parameters: 12,231
Epoch [  5/50] | Train 0.104503 | Val 0.115958 | Best 0.115958
Epoch [ 10/50] | Train 0.070330 | Val 0.078003 | Best 0.078003
Epoch [ 15/50] | Train 0.069583 | Val 0.090687 | Best 0.078003
Epoch [ 20/50] | Train 0.066424 | Val 0.084911 | Best 0.072300
Epoch [ 25/50] | Train 0.060410 | Val 0.076222 | Best 0.072300
Epoch [ 30/50] | Train 0.060069 | Val 0.071134 | Best 0.071134
Epoch [ 35/50] | Train 0.057068 | Val 0.075919 | Best 0.071134
Epoch [ 40/50] | Train 0.054682 | Val 0.080189 | Best 0.071134
Epoch [ 45/50] | Train 0.054741 | Val 0.087041 | Best 0.071134
Epoch [ 50/50] | Train 0.051917 | Val 0.083707 | Best 0.069041

=============================================
Test L2 Loss : 0.052792
Test MSE     : 0.00084303
Test MAE     : 0.00008344
Test MAPE    : 0.9122
=============================================

Gamma Statistics:
Mean Gamma   : 0.170708
Std Gamma    : 0.186399
Min Gamma    : 0.037489
Max Gamma    : 0.883611




Parameters: 12,231
Epoch [  5/50] | Train 0.119252 | Val 0.154003 | Best 0.154003
Epoch [ 10/50] | Train 0.069646 | Val 0.082332 | Best 0.082332
Epoch [ 15/50] | Train 0.063084 | Val 0.082045 | Best 0.071862
Epoch [ 20/50] | Train 0.062789 | Val 0.077574 | Best 0.068128
Epoch [ 25/50] | Train 0.068032 | Val 0.119283 | Best 0.063493
Epoch [ 30/50] | Train 0.058979 | Val 0.065429 | Best 0.063493
Epoch [ 35/50] | Train 0.061252 | Val 0.080934 | Best 0.062777
Epoch [ 40/50] | Train 0.058631 | Val 0.075279 | Best 0.062777
Epoch [ 45/50] | Train 0.058338 | Val 0.059559 | Best 0.059559
Epoch [ 50/50] | Train 0.055354 | Val 0.071402 | Best 0.059559

=============================================
Test L2 Loss : 0.050076
Test MSE     : 0.00099725
Test MAE     : 0.00008147
Test MAPE    : 0.6675
=============================================

Gamma Statistics:
Mean Gamma   : 0.185909
Std Gamma    : 0.161831
Min Gamma    : 0.040213
Max Gamma    : 0.868219



Parameters: 12,231
Epoch [  5/50] | Train 0.095193 | Val 0.095685 | Best 0.095685
Epoch [ 10/50] | Train 0.070012 | Val 0.093761 | Best 0.083846
Epoch [ 15/50] | Train 0.062901 | Val 0.096822 | Best 0.077207
Epoch [ 20/50] | Train 0.060274 | Val 0.074450 | Best 0.074450
Epoch [ 25/50] | Train 0.058764 | Val 0.093207 | Best 0.074450
Epoch [ 30/50] | Train 0.056343 | Val 0.076864 | Best 0.074450
Epoch [ 35/50] | Train 0.055568 | Val 0.084943 | Best 0.074450
Epoch [ 40/50] | Train 0.055086 | Val 0.086490 | Best 0.074450

Early stopping at epoch 40

=============================================
Test L2 Loss : 0.048391
Test MSE     : 0.00086093
Test MAE     : 0.00007717
Test MAPE    : 0.7062
=============================================

Gamma Statistics:
Mean Gamma   : 0.170334
Std Gamma    : 0.100139
Min Gamma    : 0.068916
Max Gamma    : 0.531868












# ─────────────────────────────────────────
# Config 업데이트
# ─────────────────────────────────────────
cfg = config

cfg.learning_rate = 1e-2
cfg.seed = 42
cfg.num_epochs = 50
device = cfg.device
cfg.num_layers_lstm = 2

Parameters: 4,551
Epoch [  5/50] | Train 0.128249 | Val 0.131804 | Best 0.131804
Epoch [ 10/50] | Train 0.069580 | Val 0.094553 | Best 0.075921
Epoch [ 15/50] | Train 0.062066 | Val 0.078623 | Best 0.071795
Epoch [ 20/50] | Train 0.059377 | Val 0.078745 | Best 0.071795

Early stopping at epoch 21

=============================================
Test L2 Loss : 0.050473
Test MSE     : 0.00085514
Test MAE     : 0.00007807
Test MAPE    : 0.7596
=============================================

Gamma Statistics:
Mean Gamma   : 0.327461
Std Gamma    : 0.335847
Min Gamma    : -0.008724
Max Gamma    : 1.811821






Parameters: 4,551
Epoch [  5/50] | Train 0.074754 | Val 0.072737 | Best 0.071648
Epoch [ 10/50] | Train 0.067011 | Val 0.073007 | Best 0.071648
Epoch [ 15/50] | Train 0.064279 | Val 0.076025 | Best 0.071648
Epoch [ 20/50] | Train 0.058386 | Val 0.078645 | Best 0.071648

Early stopping at epoch 24

=============================================
Test L2 Loss : 0.052922
Test MSE     : 0.00096729
Test MAE     : 0.00008074
Test MAPE    : 0.7698
=============================================

Gamma Statistics:
Mean Gamma   : 0.163463
Std Gamma    : 0.064140
Min Gamma    : 0.061083
Max Gamma    : 0.378570






learnable
Parameters: 4,551
Epoch [  5/50] | Train 0.073232 | Val 0.083126 | Best 0.078545
Epoch [ 10/50] | Train 0.071423 | Val 0.082327 | Best 0.078545
Epoch [ 15/50] | Train 0.058773 | Val 0.072156 | Best 0.071229
Epoch [ 20/50] | Train 0.057138 | Val 0.075666 | Best 0.071229
Epoch [ 25/50] | Train 0.055620 | Val 0.092581 | Best 0.071219
Epoch [ 30/50] | Train 0.058200 | Val 0.092546 | Best 0.071219
Epoch [ 35/50] | Train 0.053488 | Val 0.074877 | Best 0.071219
Epoch [ 40/50] | Train 0.050383 | Val 0.087227 | Best 0.071219
Epoch [ 45/50] | Train 0.051117 | Val 0.073517 | Best 0.070882
Epoch [ 50/50] | Train 0.109894 | Val 0.174255 | Best 0.070882

=============================================
Test L2 Loss : 0.048936
Test MSE     : 0.00080576
Test MAE     : 0.00007853
Test MAPE    : 0.8325
=============================================

Gamma Statistics:
Mean Gamma   : 0.146739
Std Gamma    : 0.068817
Min Gamma    : 0.049998
Max Gamma    : 0.407760
