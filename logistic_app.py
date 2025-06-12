import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="羅吉斯迴歸比較分析", layout="wide")
st.title("羅吉斯迴歸分析工具 | 手動選標籤與特徵 | SMOTE & Weighted 比較")

# 數據上傳/示例
st.sidebar.header("數據來源")
data_file = st.sidebar.file_uploader("上傳CSV或Excel檔", type=["csv", "xlsx"])
gen_sample = st.sidebar.button("產生示例數據")

def load_data():
    if data_file is not None:
        try:
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            return df
        except Exception as e:
            st.error(f"讀取失敗: {e}")
            return None
    elif gen_sample:
        X = np.random.normal(size=(120, 4))
        y = (X[:, 0] + 0.7 * X[:, 1] - 0.5 * X[:, 2] > 0.7).astype(int)
        df = pd.DataFrame(X, columns=["A", "B", "C", "D"])
        df["Target"] = y
        return df
    return None

df = load_data()

if df is not None:
    st.subheader("數據預覽")
    st.dataframe(df)
    # 1. 手動選標籤欄位
    label_col = st.selectbox(
        "請選擇標籤（Label/預測目標）欄位",
        df.columns
    )
    # 2. 手動多選特徵欄位（預設為全部非標籤欄）
    default_features = [col for col in df.columns if col != label_col]
    feature_cols = st.multiselect(
        "請勾選特徵（觀察項）欄位（至少一欄）",
        [col for col in df.columns if col != label_col],
        default=default_features
    )
    if not feature_cols:
        st.warning("請至少選擇一個特徵欄位。")
    else:
        st.session_state['label_col'] = label_col
        st.session_state['feature_cols'] = feature_cols
        X = df[feature_cols]
        y = df[label_col]

        # 前處理
        st.sidebar.header("前處理")
        fillna = st.sidebar.checkbox("自動補齊缺失值（中位數）", value=True)
        standardize = st.sidebar.checkbox("標準化（Standardize）")
        if fillna:
            X = X.fillna(X.median(numeric_only=True))
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # 訓練/測試分割
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        if st.button("訓練模型（SMOTE & Weighted）"):
            # y 一定要是一維 array 且不能有 nan
            y_train_1d = np.array(y_train).ravel()
            if np.any(pd.isnull(y_train_1d)):
                st.error("標籤欄位（y）有缺失值，請檢查數據！")
            else:
                try:
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X_train, y_train_1d)
                    model_smote = LogisticRegression()
                    model_smote.fit(X_res, y_res)
                    st.session_state['model_smote'] = model_smote

                    total = len(y_train_1d)
                    classes, counts = np.unique(y_train_1d, return_counts=True)
                    weights = {k: total/(2*v) for k, v in zip(classes, counts)}
                    model_weighted = LogisticRegression(class_weight=weights)
                    model_weighted.fit(X_train, y_train_1d)
                    st.session_state['model_weighted'] = model_weighted

                    st.success("模型訓練完成，可以下方比較分析。")
                    st.session_state['analyzed'] = True
                except Exception as e:
                    st.error(f"模型訓練錯誤：{e}")

if st.session_state.get('analyzed', False):
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    model_smote = st.session_state['model_smote']
    model_weighted = st.session_state['model_weighted']
    feature_cols = st.session_state['feature_cols']

    st.header("SMOTE v.s. Weighted 分析比較")
    col1, col2 = st.columns(2)
    y_pred_smote = model_smote.predict(X_test)
    y_pred_weighted = model_weighted.predict(X_test)

    with col1:
        st.markdown("### SMOTE")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.2f}")
        st.text(classification_report(y_test, y_pred_smote, zero_division=0))
        st.write("混淆矩陣")
        st.write(confusion_matrix(y_test, y_pred_smote))
    with col2:
        st.markdown("### Weighted")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_weighted):.2f}")
        st.text(classification_report(y_test, y_pred_weighted, zero_division=0))
        st.write("混淆矩陣")
        st.write(confusion_matrix(y_test, y_pred_weighted))

    # 特徵重要性
    st.subheader("特徵重要性比較（絕對值）")
    coefs = pd.DataFrame({
        "特徵": feature_cols,
        "SMOTE": np.abs(model_smote.coef_[0]),
        "Weighted": np.abs(model_weighted.coef_[0])
    })
    st.dataframe(coefs)
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(coefs))
    ax.bar(index, coefs['SMOTE'], bar_width, label='SMOTE')
    ax.bar(index+bar_width, coefs['Weighted'], bar_width, label='Weighted')
    ax.set_xticks(index+bar_width/2)
    ax.set_xticklabels(coefs['特徵'])
    ax.set_ylabel("Importance")
    ax.legend()
    st.pyplot(fig)

    # 預測新數據
    st.header("預測新數據")
    pred_mode = st.radio("預測方式", ["單筆預測", "批量預測"], horizontal=True)

    if pred_mode == "單筆預測":
        inputs = []
        cols = st.columns(len(feature_cols))
        for i, feat in enumerate(feature_cols):
            val = cols[i].number_input(feat, value=0.0, key=f"pred_{feat}")
            inputs.append(val)
        if st.button("預測這一筆"):
            arr = np.array(inputs).reshape(1, -1)
            pred_smote = model_smote.predict(arr)[0]
            prob_smote = model_smote.predict_proba(arr)[0]
            pred_weighted = model_weighted.predict(arr)[0]
            prob_weighted = model_weighted.predict_proba(arr)[0]
            st.success(f"SMOTE預測: {pred_smote}，機率: {prob_smote}")
            st.success(f"Weighted預測: {pred_weighted}，機率: {prob_weighted}")

    else:
        batch_file = st.file_uploader("上傳預測數據檔（欄位需與特徵欄完全相同）", type=["csv", "xlsx"], key="batchpred")
        if batch_file:
            if batch_file.name.endswith('.csv'):
                pred_df = pd.read_csv(batch_file)
            else:
                pred_df = pd.read_excel(batch_file)
            if list(pred_df.columns) != list(feature_cols):
                st.error(f"特徵欄位需完全相同: {feature_cols}")
            else:
                arr = pred_df.values
                pred_smote = model_smote.predict(arr)
                prob_smote = model_smote.predict_proba(arr)
                pred_weighted = model_weighted.predict(arr)
                prob_weighted = model_weighted.predict_proba(arr)
                out = pred_df.copy()
                out["SMOTE_pred"] = pred_smote
                out["Weighted_pred"] = pred_weighted
                out["SMOTE_prob_0"] = prob_smote[:,0]
                out["SMOTE_prob_1"] = prob_smote[:,1]
                out["Weighted_prob_0"] = prob_weighted[:,0]
                out["Weighted_prob_1"] = prob_weighted[:,1]
                st.dataframe(out)
                towrite = io.BytesIO()
                out.to_excel(towrite, index=False)
                towrite.seek(0)
                st.download_button("下載預測結果（Excel）", towrite, file_name="batch_predict.xlsx")

st.info("Step1. 上傳數據 → Step2. 選標籤與特徵 → Step3. 訓練模型 → Step4. 下方預測或下載結果")
