#type:ignore

Parameters: 14,975
/usr/local/lib/python3.12/dist-packages/torchdiffeq/_impl/misc.py:306: UserWarning: t is not on the same device as y0. Coercing to y0.device.
  warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
Epoch [  2/50] | Train 0.094693 | Val 0.058790 | Best Train 0.094693 | Best Val 0.058790
Epoch [  4/50] | Train 0.047381 | Val 0.044577 | Best Train 0.047381 | Best Val 0.044577
Epoch [  6/50] | Train 0.041700 | Val 0.036973 | Best Train 0.041700 | Best Val 0.036973
Epoch [  8/50] | Train 0.039446 | Val 0.046969 | Best Train 0.039446 | Best Val 0.036973
Epoch [ 10/50] | Train 0.039644 | Val 0.036318 | Best Train 0.039446 | Best Val 0.036318
Epoch [ 12/50] | Train 0.038734 | Val 0.038199 | Best Train 0.038734 | Best Val 0.036318
Epoch [ 14/50] | Train 0.037126 | Val 0.040560 | Best Train 0.037126 | Best Val 0.036095
Epoch [ 16/50] | Train 0.037316 | Val 0.034370 | Best Train 0.037126 | Best Val 0.034370
Epoch [ 18/50] | Train 0.038553 | Val 0.036143 | Best Train 0.037126 | Best Val 0.034370
Epoch [ 20/50] | Train 0.035925 | Val 0.038337 | Best Train 0.035925 | Best Val 0.034370
Epoch [ 22/50] | Train 0.036912 | Val 0.035532 | Best Train 0.035925 | Best Val 0.034370
Epoch [ 24/50] | Train 0.036346 | Val 0.035166 | Best Train 0.035925 | Best Val 0.034370

Early stopping at epoch 24

=============================================
  Val-Best Checkpoint
=============================================
Test L2 Loss : 0.047193
Test MSE     : 0.00092113
Test MAE     : 0.00007809
Test MAPE    : 0.4772
=============================================




Parameters: 5,319
Epoch [  2/50] | Train 0.174674 | Val 0.126820 | Best Train 0.174674 | Best Val 0.126820
Epoch [  4/50] | Train 0.116464 | Val 0.098249 | Best Train 0.116464 | Best Val 0.098249
Epoch [  6/50] | Train 0.095734 | Val 0.085976 | Best Train 0.095734 | Best Val 0.085976
Epoch [  8/50] | Train 0.042767 | Val 0.039166 | Best Train 0.042767 | Best Val 0.039166
Epoch [ 10/50] | Train 0.040946 | Val 0.036085 | Best Train 0.040946 | Best Val 0.036085
Epoch [ 12/50] | Train 0.041187 | Val 0.039886 | Best Train 0.040946 | Best Val 0.036085
Epoch [ 14/50] | Train 0.040212 | Val 0.035485 | Best Train 0.040212 | Best Val 0.035485
Epoch [ 16/50] | Train 0.040380 | Val 0.036184 | Best Train 0.038466 | Best Val 0.035485
Epoch [ 18/50] | Train 0.039034 | Val 0.033412 | Best Train 0.038466 | Best Val 0.033412
Epoch [ 20/50] | Train 0.039344 | Val 0.041626 | Best Train 0.038466 | Best Val 0.033412
Epoch [ 22/50] | Train 0.038198 | Val 0.044677 | Best Train 0.037247 | Best Val 0.033412
Epoch [ 24/50] | Train 0.037379 | Val 0.036347 | Best Train 0.037247 | Best Val 0.033412
Epoch [ 26/50] | Train 0.037092 | Val 0.039173 | Best Train 0.037092 | Best Val 0.033412

Early stopping at epoch 26

=============================================
  Val-Best Checkpoint
=============================================
Test L2 Loss : 0.046623
Test MSE     : 0.00085930
Test MAE     : 0.00007704
Test MAPE    : 0.4604
=============================================




cfg.learning_rate = 4.5e-3
cfg.seed = 42
cfg.num_epochs = 15
device = cfg.device
cfg.num_layers_lstm = 2
cfg.hidden_dim = 10
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 16
cfg.lambda_qlike = 0.0
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 2.2




Parameters: 5,319
Epoch [  2/15] | Train 0.174674 | Val 0.126820 | Best Train 0.174674 | Best Val 0.126820
Epoch [  4/15] | Train 0.116464 | Val 0.098249 | Best Train 0.116464 | Best Val 0.098249
Epoch [  6/15] | Train 0.095734 | Val 0.085976 | Best Train 0.095734 | Best Val 0.085976
Epoch [  8/15] | Train 0.042767 | Val 0.039166 | Best Train 0.042767 | Best Val 0.039166
Epoch [ 10/15] | Train 0.040946 | Val 0.036085 | Best Train 0.040946 | Best Val 0.036085
Epoch [ 12/15] | Train 0.041187 | Val 0.039886 | Best Train 0.040946 | Best Val 0.036085
Epoch [ 14/15] | Train 0.040212 | Val 0.035485 | Best Train 0.040212 | Best Val 0.035485

=============================================
  Val-Best Checkpoint
=============================================
Test L2 Loss : 0.045584
Test MSE     : 0.00073021
Test MAE     : 0.00007245
Test MAPE    : 0.5212
=============================================








cfg.learning_rate = 0.0095795680261937
cfg.seed = 42
cfg.num_epochs = 10
device = cfg.device
cfg.num_layers_lstm = 3
cfg.hidden_dim = 12
cfg.lstm_hidden_dim = cfg.hidden_dim
cfg.ode_hidden_dim = 16
cfg.lambda_qlike = 0.0
cfg.use_learnable_exp = False
cfg.use_non_learnable_exp = True
cfg.exp_deno_init = 13.568876566846562




Parameters: 10,729
Epoch [  2/10] | Val 0.055389 | Best Val 0.055389
Epoch [  4/10] | Val 0.046820 | Best Val 0.046820
Epoch [  6/10] | Val 0.048081 | Best Val 0.042250
Epoch [  8/10] | Val 0.040211 | Best Val 0.038062
Epoch [ 10/10] | Val 0.036012 | Best Val 0.036012






=============================================
**************DLQF RNN WITH ODE**************
=============================================
Test L2 Loss : 0.048534
Test MSE     : 0.00070665
Test MAE     : 0.00007630
Test MAPE    : 0.4983
=============================================





# ─────────────────────────────────────────
# Load best model & evaluate on test set
# ─────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()


test_loss_list = []
rv_preds, rv_trues, gammas = [], [], []


with torch.no_grad():
    for i in range(len(X_te)):
        x_b = X_te[i]
        y_b = Y_te[i]
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        r_q, gamma = model(x_b, alpha)
        test_loss = l2_distance_loss(y_b, r_q)
        test_loss_1d = test_loss.unsqueeze(-1)

        test_loss_list.append(test_loss_1d.cpu())

        rv_pred_unscaled = (r_q ** 2).sum(dim=1) + gamma
        rv_true_unscaled = (y_b ** 2).sum(dim=1)

        rv_pred = rv_pred_unscaled / (cfg.scale_factor ** 2)
        rv_true = rv_true_unscaled / (cfg.scale_factor ** 2)

        rv_pred_1d = rv_pred.unsqueeze(-1)
        rv_true_1d = rv_true.unsqueeze(-1)
        gamma_1d = gamma.unsqueeze(-1)

        rv_preds.append(rv_pred_1d.cpu())
        rv_trues.append(rv_true_1d.cpu())
        gammas.append(gamma_1d.cpu())



test_losses = torch.cat(test_loss_list).mean(dim=0).item()
rv_preds = torch.cat(rv_preds)
rv_trues = torch.cat(rv_trues)
gammas = torch.cat(gammas)

mse  = ((rv_preds - rv_trues) ** 2).mean(dim=0).item()
mae  = (rv_preds - rv_trues).abs().mean(dim=0).item()
mape = ((rv_preds - rv_trues).abs() / (rv_trues + 1e-8)).mean(dim=0).item()


df = pd.DataFrame({
    'L2': test_losses.squeeze().cpu().numpy(),
    'MSE': mse.squeeze().cpu().numpy(),
    'MAE': mae.squeeze().cpu().numpy(),
    'MAPE': mape.squeeze().cpu().numpy()
})

# Transpose해서 row = metric, column = 데이터 포인트
df = df.T

# Column names: 1, 2, 3, ... len(X_te)
df.columns = range(1, df.shape[1] + 1)


print(f"\n{'='*45}")
print('*****************DLQF RNN******************')
print(f"{'='*45}\n")
df