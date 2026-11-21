import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')

def load_data(path):

    # Đọc file
    data_frame = pd.read_csv(path)

    # Kiểm tra và xử lý nếu có giá trị thiếu (nếu có thì xóa)
    if data_frame.isnull().sum().sum() > 0:
        data_frame = data_frame.dropna()

    # Đặt input là feature_names
    feature_names = ['TV', 'Radio', 'Newspaper']

    # 4. Chuyển sang mảng NumPy (float) để đồng bộ tất cả dữ liệu trong file.csv
    x = data_frame[feature_names].to_numpy(dtype=float)
    y = data_frame['Sales'].to_numpy(dtype=float)

    return x, y, feature_names


# ==============================================================================
# PHẦN 2: PHÂN CHIA VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Chia du lieu thanh tap train va test theo ty le 8:2.
    """
    X_train, X_test, y_train, y_test = sklearn_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)
    return mean, std


def transform(X, mean, std):
    X_scaled = (X - mean) / std
    return X_scaled


def add_bias(X):
    m = X.shape[0]
    bias = np.ones((m, 1))
    Xb = np.hstack((bias, X))
    return Xb


# ==============================================================================
# PHẦN 3: HÀM DỰ ĐOÁN VÀ HÀM MẤT MÁT
# ==============================================================================
def predict(Xb, w):
    return Xb @ w  # @ là phép nhân ma trận


def compute_cost(Xb, y, w):
    m = len(y)  # m = số mẫu
    err = predict(Xb, w) - y  # sai số của từng mẫu: y_pred - y
    return float(err @ err / (2 * m))  # err @ err = tổng bình phương sai số


# ==============================================================================
# PHẦN 4: HUẤN LUYỆN BẰNG GRADIENT DESCENT
def train(Xb, y, lr, n_iters=1000):
    m, n = Xb.shape
    w = np.zeros(n)
    history = []

    for _ in range(n_iters):
        cost = compute_cost(Xb, y, w)  # J(w) với w hiện tại
        history.append(cost)

        grad = (Xb.T @ (predict(Xb, w) - y)) / m
        w = w - lr * grad

        if not np.isfinite(cost):  # lr quá lớn thì dừng
            break

    return w, history


# ==============================================================================
# PHẦN 5: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM TRA 
# ==============================================================================

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)


def evaluate(y_true, y_pred):
    results = {
        "MSE": float(mse(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    return results


# ==============================================================================
# PHẦN 6: GHÉP PIPELINE & THỬ NGHIỆM TỐC ĐỘ HỌC 
# ==============================================================================
def run_pipeline(path, learning_rates, n_iters=1000):
    # 1. Đọc dữ liệu
    X, y, feature_names = load_data(path)

    # 2. Chia dữ liệu 8:2 và tiền xử lý (mean, std chỉ tính trên tập train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    mean, std = fit_scaler(X_train)
    Xb_train = add_bias(transform(X_train, mean, std))
    Xb_test = add_bias(transform(X_test, mean, std))

    # 3. Huấn luyện và đánh giá với từng learning rate
    results = []
    histories = {}
    weights = {}

    for lr in learning_rates:
        w, history = train(Xb_train, y_train, lr=lr, n_iters=n_iters)
        histories[lr] = history
        weights[lr] = w
        final_cost = history[-1]

        # Kiểm tra mô hình có hội tụ không
        if np.isfinite(final_cost) and len(history) == n_iters:
            metrics = evaluate(y_test, predict(Xb_test, w))
            results.append({
                "Learning rate": lr,
                "Final Cost": round(final_cost, 6),
                "MSE (Test)": round(metrics["MSE"], 6),
                "RMSE (Test)": round(metrics["RMSE"], 6),
                "R² (Test)": round(metrics["R2"], 6)
            })
        else:
            results.append({
                "Learning rate": lr,
                "Final Cost": "Phân kỳ",
                "MSE (Test)": "-",
                "RMSE (Test)": "-",
                "R² (Test)": "-"
            })

    # 4. Chọn best_lr theo Final Cost trên tập train
    # Nếu nhiều lr có cost thấp nhất gần bằng nhau thì chọn lr nhỏ nhất cho ổn định
    valid = {lr: h[-1] for lr, h in histories.items()
             if len(h) == n_iters and np.isfinite(h[-1])}
    min_cost = min(valid.values())
    best_lr = min(lr for lr, c in valid.items() if c <= min_cost + 1e-7)
    best_w = weights[best_lr]

    # 5. Vẽ biểu đồ đường cong hội tụ cost (cost_curves.png)
    plt.figure(figsize=(10, 6))
    for lr in valid:
        plt.plot(range(n_iters), histories[lr], label=f"lr = {lr}")

    plt.title("Đường cong hàm mất mát J(w) qua 1000 vòng lặp")
    plt.xlabel("Số vòng lặp (Iterations)")
    plt.ylabel("Cost J(w)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("cost_curves.png", dpi=300)

    # 6. In bảng kết quả
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("BẢNG KẾT QUẢ THỬ NGHIỆM TỐC ĐỘ HỌC")
    print("=" * 65)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 65)
    print(f"Tốc độ học tốt nhất (best_lr): {best_lr}")
    print(f"Trọng số [w0, w_TV, w_Radio, w_Newspaper]: {np.round(best_w, 4)}")
    print("=" * 65)

    # 7. Đối chiếu với scikit-learn
    sk_model = LinearRegression(fit_intercept=False)
    sk_model.fit(Xb_train, y_train)
    sk_metrics = evaluate(y_test, sk_model.predict(Xb_test))
    print("\nĐỐI CHIẾU SKLEARN LINEAR REGRESSION:")
    print(f"MSE : {sk_metrics['MSE']:.6f}")
    print(f"RMSE: {sk_metrics['RMSE']:.6f}")
    print(f"R²  : {sk_metrics['R2']:.6f}")

    plt.show()  # để cuối hàm để không chặn việc in kết quả
    return df_results, best_lr, best_w


# ==============================================================================
# PHẦN BỔ SUNG: VẼ BIỂU ĐỒ THỰC TẾ VS DỰ ĐOÁN (linear_fit_plot.png)
# ==============================================================================
def plot_regression_fit(path, best_w):
    X, y, _ = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    mean, std = fit_scaler(X_train)
    Xb_test = add_bias(transform(X_test, mean, std))

    plt.figure(figsize=(6.5, 5.5))

    # Thực tế vs Dự đoán trên tập test (3 biến)
    y_test_pred = predict(Xb_test, best_w)
    plt.scatter(y_test, y_test_pred, color="seagreen", edgecolors="black", alpha=0.8, s=60, label="Điểm dự đoán (Test)")
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="darkorange", linestyle="--", linewidth=2, label="Đường lý tưởng (y_pred = y_test)")
    plt.title("Độ khớp mô hình trên tập Test (3 biến)")
    plt.xlabel("Doanh số thực tế (y_test)")
    plt.ylabel("Doanh số dự đoán (y_pred)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Ghi các chỉ số lên hình, đặt tỉ lệ 2 trục bằng nhau để đường chéo đúng 45 độ
    kq = evaluate(y_test, y_test_pred)
    plt.text(0.97, 0.05, f"MSE = {kq['MSE']:.4f}\nRMSE = {kq['RMSE']:.4f}\nR² = {kq['R2']:.4f}",
             transform=plt.gca().transAxes, ha="right", va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("linear_fit_plot.png", dpi=300)
    plt.show()


# ==============================================================================
# PHẦN BỔ SUNG: VẼ 3 ĐƯỜNG HỒI QUY TUYẾN TÍNH (TV, Radio, Newspaper vs Sales)
# ==============================================================================
def plot_3_linear_lines(path):
    df = pd.read_csv(path)
    y = df["Sales"].to_numpy(dtype=float)
    names = ["TV", "Radio", "Newspaper"]
    colors = ["royalblue", "seagreen", "darkorange"]

    plt.figure(figsize=(15, 4.8))
    for k, (name, c) in enumerate(zip(names, colors)):
        x = df[name].to_numpy(dtype=float)
        Xb = add_bias(x.reshape(-1, 1))
        w = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        y_pred = predict(Xb, w)
        r2 = r2_score(y, y_pred)

        plt.subplot(1, 3, k + 1)
        plt.scatter(x, y, color=c, alpha=0.65, label="Điểm dữ liệu thực tế")
        xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, w[0] + w[1] * xs, color="crimson", linewidth=2.5,
                 label=f"y = {w[0]:.3f} + {w[1]:.4f}·x")
        plt.title(f"Hồi quy tuyến tính: {name} vs Sales")
        plt.xlabel(f"Chi phí quảng cáo {name}")
        plt.ylabel("Doanh số (Sales)")
        plt.text(0.97, 0.05, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("linear_3_features.png", dpi=300)
    plt.show()


# ==============================================================================
# KHỐI CHẠY CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == "__main__":
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    df_results, best_lr, best_w = run_pipeline(
        path="advertising.csv",
        learning_rates=learning_rates,
        n_iters=1000
    )

    # Vẽ thêm biểu đồ thực tế vs dự đoán trên tập test
    plot_regression_fit("advertising.csv", best_w)

    # Vẽ 3 đường hồi quy tuyến tính của TV, Radio, Newspaper
    plot_3_linear_lines("advertising.csv")