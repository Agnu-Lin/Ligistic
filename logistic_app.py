import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io

# ========= 語系包 =========
lang_dict = {
    "zh": {
        "title": "智能機器學習分析平台",
        "upload": "請上傳數據（CSV 或 Excel）",
        "sample": "產生示例數據",
        "choose_label": "選擇標籤（目標欄位）",
        "choose_model": "選擇模型",
        "choose_sampling": "選擇不平衡處理方式",
        "smote": "SMOTE過採樣",
        "weight": "類別加權",
        "none": "不處理",
        "preprocess": "資料預處理",
        "standardize": "標準化特徵",
        "fillna": "自動補缺失值",
        "run": "開始分析",
        "preview": "數據預覽",
        "analyze": "分析報告",
        "download_report": "下載 Markdown 報告",
        "download_pred": "下載預測結果",
        "predict_mode": "預測方式",
        "single_pred": "單筆預測",
        "batch_pred": "批量預測",
        "feature_importance": "特徵重要性（長條圖）",
        "learning_curve": "學習曲線",
        "confusion": "混淆矩陣",
        "decision_boundary": "決策邊界(僅2D)",
        "save_model": "儲存模型",
        "load_model": "載入模型",
        "choose_file": "選擇檔案",
        "choose_model_file": "選擇已訓練模型檔",
        "login": "用戶登入（進階功能）"
    },
    "en": {
        "title": "Smart ML Analysis Platform",
        "upload": "Upload data (CSV or Excel)",
        "sample": "Generate sample data",
        "choose_label": "Select label (target column)",
        "choose_model": "Select model",
        "choose_sampling": "Select imbalance treatment",
        "smote": "SMOTE oversampling",
        "weight": "Class weighting",
        "none": "None",
        "preprocess": "Preprocessing",
        "standardize": "Standardize features",
        "fillna": "Fill missing values",
        "run": "Start analysis",
        "preview": "Data preview",
        "analyze": "Analysis report",
        "download_report": "Download Markdown report",
        "download_pred": "Download prediction",
        "predict_mode": "Prediction mode",
        "single_pred": "Single prediction",
        "batch_pred": "Batch prediction",
        "feature_importance": "Feature Importance (Bar)",
        "learning_curve": "Learning curve",
        "confusion": "Confusion Matrix",
        "decision_boundary": "Decision boundary (2D only)",
        "save_model": "Save model",
        "load_model": "Load model",
        "choose_file": "Choose file",
        "choose_model_file": "Choose trained model",
        "login": "User login (advanced)"
    }
}

# ============ UI 基本設定 ============
lang = st.sidebar.selectbox("Language / 語言", ["中文", "English"])
LANG = "zh" if lang == "中文" else "en"
L = lang_dict[LANG]
st.title(L["title"])

# ======== 數據上傳與標籤選擇 =========
st.subheader(L["upload"])
data_file = st.file_uploader(L["choose_file"], type=["csv", "xlsx"])
if st.button(L["sample"]):
    np.random.seed(0)
    X = np.random.normal(size=(150, 3))
    y = np.random.choice([0,1,2], size=150)
    df = pd.DataFrame(X, columns=["Feature1","Feature2","Feature3"])
    df["target"] = y
else:
    df = None
if data_file:
    try:
        if data_file.name.endswith(".csv"):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
    except Exception as e:
        st.error(str(e))
if df is not None:
    st.dataframe(df.head())
    label_col = st.selectbox(L["choose_label"], df.columns)
    X = df.drop(label_col, axis=1)
    y = df[label_col]
else:
    X = y = None

# ====== 資料預處理選項 ======
st.subheader(L["preprocess"])
standardize = st.checkbox(L["standardize"])
fillna = st.checkbox(L["fillna"])

# ========== 模型選擇 ==========
st.subheader(L["choose_model"])
model_name = st.selectbox("", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "XGBoost"])
sampling = st.radio(L["choose_sampling"], [L["none"], L["smote"], L["weight"]])

# ======= 開始分析按鈕 =======
if st.button(L["run"]) and X is not None:
    # 資料預處理
    if fillna:
        X = X.fillna(X.median(numeric_only=True))
    if standardize:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # 資料集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # SMOTE
    if sampling == L["smote"]:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    # 選擇模型
    if model_name == "Logistic Regression":
        model = LogisticRegression(class_weight="balanced" if sampling==L["weight"] else None, max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(class_weight="balanced" if sampling==L["weight"] else None)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(class_weight="balanced" if sampling==L["weight"] else None, n_estimators=100)
    elif model_name == "SVM":
        model = SVC(probability=True, class_weight="balanced" if sampling==L["weight"] else None)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"{L['analyze']}")
    st.write(classification_report(y_test, y_pred, zero_division=0))
    st.write(f"{L['confusion']}")
    st.write(confusion_matrix(y_test, y_pred))
    # 特徵重要性
    st.write(L["feature_importance"])
    if hasattr(model, "feature_importances_"):
        feat_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        feat_imp = np.abs(model.coef_[0])
    else:
        feat_imp = np.zeros(X.shape[1])
    fig, ax = plt.subplots()
    ax.barh(X.columns, feat_imp)
    st.pyplot(fig)
    # 學習曲線
    st.write(L["learning_curve"])
    fig2, ax2 = plt.subplots()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1,1,5))
    ax2.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    ax2.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation")
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)
    # 決策邊界（僅2D顯示）
    if X.shape[1]==2:
        st.write(L["decision_boundary"])
        fig3, ax3 = plt.subplots()
        x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
        y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax3.contourf(xx, yy, Z, alpha=0.4)
        ax3.scatter(X.iloc[:,0], X.iloc[:,1], c=y)
        st.pyplot(fig3)
    # 預測單筆/批量
    st.subheader(L["predict_mode"])
    mode = st.radio("", [L["single_pred"], L["batch_pred"]])
    if mode == L["single_pred"]:
        pred_vals = [st.number_input(f, key=f"p_{f}") for f in X.columns]
        if st.button("預測/Prediction"):
            arr = np.array(pred_vals).reshape(1,-1)
            pred = model.predict(arr)[0]
            prob = model.predict_proba(arr)[0]
            st.write(f"預測類別: {pred}，機率分布: {prob}")
    else:
        file_pred = st.file_uploader("上傳批次預測檔", type=["csv","xlsx"], key="predf")
        if file_pred:
            if file_pred.name.endswith(".csv"):
                pred_df = pd.read_csv(file_pred)
            else:
                pred_df = pd.read_excel(file_pred)
            pred_vals = pred_df.values
            pred = model.predict(pred_vals)
            prob = model.predict_proba(pred_vals)
            out = pred_df.copy()
            out["pred"] = pred
            for i in range(prob.shape[1]):
                out[f"prob_{i}"] = prob[:,i]
            towrite = io.BytesIO()
            out.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(L["download_pred"], towrite, file_name="pred_results.xlsx")
    # 分析報告
    md_report = f"# {L['analyze']}\n\n"
    md_report += f"## {L['choose_model']}: {model_name}\n\n"
    md_report += f"### {L['confusion']}\n```\n{confusion_matrix(y_test, y_pred)}\n```\n"
    md_report += f"### {L['feature_importance']}\n{list(feat_imp)}\n"
    md_report += f"### Report\n```\n{classification_report(y_test, y_pred, zero_division=0)}\n```\n"
    st.download_button(L["download_report"], md_report, file_name="report.md")

# (模型儲存/載入與 API/Security、Google Sheets整合可進階補加)
st.info("進階功能如 Google Sheets、自動 API、權限安全等，可根據需求補充。這是全功能雛形，如要再加請告知！")
