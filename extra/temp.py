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