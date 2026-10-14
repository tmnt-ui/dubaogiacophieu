
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CAU HINH TRANG WEB
# =========================
st.set_page_config(
    page_title="Du bao gia co phieu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 DU BAO GIA CO PHIEU BANG LSTM")
st.write(
    "Ung dung su dung du lieu gia dong cua tu Yahoo Finance "
    "de huan luyen mo hinh LSTM va du bao cac phien tiep theo."
)

st.info(
    "Ket qua chi phuc vu hoc tap va tham khao, "
    "khong phai loi khuyen dau tu."
)

# =========================
# HAM TAO DU LIEU CHUOI
# =========================
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


# =========================
# NHAP THONG TIN
# =========================
st.sidebar.header("Cau hinh du bao")

ticker = st.sidebar.text_input(
    "Ma co phieu Yahoo Finance",
    value="VNM.VN"
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngay bat dau lay du lieu",
    value=date(2018, 1, 1),
    max_value=date.today() - timedelta(days=61)
)

epochs = st.sidebar.slider(
    "So vong huan luyen (epochs)",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

forecast_days = st.sidebar.slider(
    "So phien can du bao",
    min_value=1,
    max_value=30,
    value=10
)

run = st.button(
    "🚀 CHAY DU BAO",
    type="primary",
    use_container_width=True
)

# =========================
# CHAY MO HINH
# =========================
if run:
    if not ticker:
        st.error("Vui long nhap ma co phieu.")
        st.stop()

    if start_date >= date.today() - timedelta(days=60):
        st.error("Ngay bat dau qua gan. Hay chon ngay som hon.")
        st.stop()

    with st.spinner("Dang tai du lieu co phieu..."):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=date.today() + timedelta(days=1),
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                st.error(
                    "Khong tim thay du lieu. "
                    "Hay kiem tra ma co phieu, vi du VNM.VN."
                )
                st.stop()

            # Xu ly cot du lieu neu Yahoo Finance tra ve MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                close_data = df["Close"]
                if isinstance(close_data, pd.DataFrame):
                    close_data = close_data.iloc[:, 0]
            else:
                close_data = df["Close"]

            prices = pd.DataFrame({
                "Date": df.index,
                "Close": pd.to_numeric(
                    close_data, errors="coerce"
                ).to_numpy().reshape(-1)
            })

            prices = prices.dropna().reset_index(drop=True)

        except Exception as e:
            st.error(f"Loi khi tai du lieu: {e}")
            st.stop()

    if len(prices) < 100:
        st.error(
            "Du lieu chua du de huan luyen mo hinh. "
            "Hay chon khoang thoi gian dai hon."
        )
        st.stop()

    st.success(
        f"Da tai {len(prices)} phien du lieu cua {ticker}."
    )

    # Chia tap huan luyen va tap kiem tra theo thu tu thoi gian
    data = prices["Close"].values.reshape(-1, 1)
    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    all_scaled = scaler.transform(data)

    window_size = 60

    X_train, y_train = create_sequences(train_scaled, window_size)

    # Tap kiem tra su dung 60 phien truoc do lam ngu canh
    test_combined = all_scaled[train_size - window_size:]
    X_test, y_test = create_sequences(test_combined, window_size)

    X_train = X_train.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    # Tach tap validation tu cuoi tap huan luyen
    val_size = max(1, int(len(X_train) * 0.1))

    X_fit = X_train[:-val_size]
    y_fit = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    if len(X_fit) == 0:
        st.error("Khong du du lieu de huan luyen.")
        st.stop()

    # Tao mo hinh LSTM
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    progress = st.progress(0, text="Dang huan luyen mo hinh LSTM...")

    try:
        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        progress.progress(100, text="Huan luyen hoan tat!")

    except Exception as e:
        st.error(f"Loi khi huan luyen mo hinh: {e}")
        st.stop()

    # =========================
    # DANH GIA MO HINH
    # =========================
    test_pred_scaled = model.predict(X_test, verbose=0)

    test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
    mae = mean_absolute_error(y_test_real, test_pred)

    if len(y_test_real) >= 2:
        r2 = r2_score(y_test_real, test_pred)
    else:
        r2 = np.nan

    last_price = float(data[-1, 0])

    # =========================
    # DU BAO CAC PHIEN TIEP THEO
    # =========================
    last_60 = all_scaled[-window_size:].reshape(
        1, window_size, 1
    )

    future_scaled = []

    for _ in range(forecast_days):
        next_pred = model.predict(last_60, verbose=0)
        next_value = float(next_pred[0, 0])
        future_scaled.append(next_value)

        last_60 = np.concatenate(
            [
                last_60[:, 1:, :],
                np.array(next_value).reshape(1, 1, 1)
            ],
            axis=1
        )

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    # Tao ngay du bao theo ngay lam viec thu 2 den thu 6
    last_date = pd.Timestamp(prices["Date"].iloc[-1])
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    # =========================
    # HIEN THI KET QUA
    # =========================
    st.header("1. Tong quan co phieu")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Gia dong cua gan nhat",
        f"{last_price:,.0f}"
    )

    col2.metric(
        f"Gia du bao phien thu {forecast_days}",
        f"{future_prices[-1]:,.0f}",
        f"{future_prices[-1] - last_price:+,.0f}"
    )

    change_pct = (
        (future_prices[-1] / last_price - 1) * 100
        if last_price != 0 else 0
    )

    col3.metric(
        "Muc thay doi du bao",
        f"{change_pct:+.2f}%"
    )

    st.caption(
        "Gia hien thi theo don vi du lieu Yahoo Finance. "
        "Ngay du bao la cac ngay lam viec uoc tinh, "
        "khong phai lich giao dich chinh thuc."
    )

    st.header("2. Bieu do gia thuc te va gia du bao")

    fig, ax = plt.subplots(figsize=(12, 5))

    recent = min(120, len(prices))

    ax.plot(
        prices["Date"].iloc[-recent:],
        prices["Close"].iloc[-recent:],
        label="Gia dong cua thuc te"
    )

    ax.plot(
        future_dates,
        future_prices,
        marker="o",
        linestyle="--",
        label="Gia du bao"
    )

    ax.set_xlabel("Ngay")
    ax.set_ylabel("Gia dong cua")
    ax.set_title(f"Du bao gia co phieu {ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    st.header("3. Bang gia du bao")

    forecast_table = pd.DataFrame({
        "Phien du bao": range(1, forecast_days + 1),
        "Ngay du kien": future_dates.strftime("%Y-%m-%d"),
        "Gia du bao": np.round(future_prices, 2)
    })

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )

    st.header("4. Danh gia mo hinh")

    m1, m2, m3 = st.columns(3)

    m1.metric("RMSE", f"{rmse:,.2f}")
    m2.metric("MAE", f"{mae:,.2f}")
    m3.metric(
        "R²",
        f"{r2:.4f}" if np.isfinite(r2) else "N/A"
    )

    st.caption(
        "RMSE va MAE do sai so du bao tren tap kiem tra; "
        "gia tri cang nho thuong cang tot. R² co the am "
        "neu mo hinh du bao kem hon moc so sanh trung binh."
    )

    st.header("5. Qua trinh huan luyen")

    loss_fig, loss_ax = plt.subplots(figsize=(10, 4))

    loss_ax.plot(
        history.history["loss"],
        label="Train Loss"
    )
    loss_ax.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Train va Validation")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    st.pyplot(loss_fig)
    plt.close(loss_fig)

    st.success("Da hoan thanh qua trinh du bao!")

else:
    st.subheader("Huong dan su dung")
    st.write(
        "1. Nhap ma co phieu o thanh ben trai.\n"
        "2. Chon ngay bat dau lay du lieu.\n"
        "3. Chon so vong huan luyen va so phien du bao.\n"
        "4. Nhan CHAY DU BAO va cho mo hinh xu ly."
    )

    st.warning(
        "Lan chay dau tien co the mat vai phut do can "
        "tai du lieu va huan luyen mo hinh."
    )
