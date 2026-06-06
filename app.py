import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


def run_pipeline(fd):
    # Processing pipeline (data contents are not displayed)

    # --- 2. Preprocessing ---
    if 'SK_ID_CURR' in fd.columns:
        fd['SK_ID_CURR'] = fd['SK_ID_CURR'].astype(str)

    all_object_cols = fd.select_dtypes(include='object').columns.tolist()
    if 'SK_ID_CURR' in all_object_cols:
        all_object_cols.remove('SK_ID_CURR')

    if len(all_object_cols) > 0:
        fd = pd.get_dummies(fd, columns=all_object_cols, drop_first=True)

    # Impute missing values
    numeric_cols = fd.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col != 'TARGET' and fd[col].isnull().any():
            fd[col] = fd[col].fillna(fd[col].median())

    cat_cols = fd.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        if fd[col].isnull().any():
            fd[col] = fd[col].fillna(fd[col].mode()[0])

    # Drop columns with >70% missing (after imputation this is unlikely)
    threshold = 0.70
    missing_percentages = fd.isnull().sum() / len(fd)
    columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    if columns_to_drop:
        fd.drop(columns=columns_to_drop, axis=1, inplace=True)

    # --- 3. Feature Engineering ---
    if 'AMT_GOODS_PRICE' in fd.columns:
        fd['AMT_GOODS_PRICE'] = fd['AMT_GOODS_PRICE'].fillna(fd['AMT_GOODS_PRICE'].median())
    else:
        fd['AMT_GOODS_PRICE'] = 0

    # Safe ratios (avoid division by zero)
    fd['CREDIT_INCOME_RATIO'] = fd.get('AMT_CREDIT', 0) / fd.get('AMT_INCOME_TOTAL', 1)
    fd['ANNUITY_INCOME_RATIO'] = fd.get('AMT_ANNUITY', 0) / fd.get('AMT_INCOME_TOTAL', 1)
    fd['ANNUITY_CREDIT_RATIO'] = fd.get('AMT_ANNUITY', 0) / fd.get('AMT_CREDIT', 1)
    fd['GOODS_PRICE_CREDIT_RATIO'] = np.where(fd.get('AMT_CREDIT', 0) != 0, fd['AMT_GOODS_PRICE'] / fd.get('AMT_CREDIT', 1), 0)

    # Age features
    if 'DAYS_BIRTH' in fd.columns:
        fd['AGE_YEARS'] = abs(fd['DAYS_BIRTH'] / 365.25)
    else:
        fd['AGE_YEARS'] = 0

    if 'DAYS_EMPLOYED' in fd.columns:
        fd['YEARS_EMPLOYED'] = abs(fd['DAYS_EMPLOYED'] / 365.25)
        fd['YEARS_EMPLOYED'] = fd['YEARS_EMPLOYED'].replace({365243: 0})
        fd['DAYS_EMPLOYED_PERC'] = fd['YEARS_EMPLOYED'] / fd['AGE_YEARS'].replace({0: np.nan})
        fd['DAYS_EMPLOYED_PERC'] = fd['DAYS_EMPLOYED_PERC'].fillna(0)
    else:
        fd['YEARS_EMPLOYED'] = 0
        fd['DAYS_EMPLOYED_PERC'] = 0

    if 'CNT_FAM_MEMBERS' in fd.columns:
        fd['CNT_FAM_MEMBERS'] = fd['CNT_FAM_MEMBERS'].fillna(fd['CNT_FAM_MEMBERS'].median())
        if 'CNT_CHILDREN' in fd.columns:
            fd['CNT_NON_CHILDREN'] = fd['CNT_FAM_MEMBERS'] - fd['CNT_CHILDREN']
            fd['CHILDREN_RATIO'] = np.where(fd['CNT_FAM_MEMBERS'] != 0, fd['CNT_CHILDREN'] / fd['CNT_FAM_MEMBERS'], 0)
        else:
            fd['CNT_NON_CHILDREN'] = 0
            fd['CHILDREN_RATIO'] = 0
    else:
        fd['CNT_NON_CHILDREN'] = 0
        fd['CHILDREN_RATIO'] = 0

    if 'CNT_FAM_MEMBERS' in fd.columns:
        fd['INCOME_PER_PERSON'] = np.where(fd['CNT_FAM_MEMBERS'] != 0, fd.get('AMT_INCOME_TOTAL', 0) / fd['CNT_FAM_MEMBERS'], 0)
    else:
        fd['INCOME_PER_PERSON'] = 0

    ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    for col in ext_source_cols:
        if col in fd.columns:
            fd[col] = fd[col].fillna(fd[col].median())
        else:
            fd[col] = 0.0

    if all(col in fd.columns for col in ext_source_cols):
        fd['EXT_SOURCES_PROD'] = fd['EXT_SOURCE_1'] * fd['EXT_SOURCE_2'] * fd['EXT_SOURCE_3']
    else:
        fd['EXT_SOURCES_PROD'] = 0

    fd['DAYS_EMPLOYED_SQ'] = fd.get('DAYS_EMPLOYED', 0) ** 2

    # feature engineering done; not displaying dataframe contents

    # --- 4. Prepare Data for Modeling ---
    if 'TARGET' not in fd.columns:
        st.error("The dataset does not contain a 'TARGET' column. Cannot train without labels.")
        return

    X = fd.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
    Y = fd['TARGET']

    # Coerce non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

    # --- 5. Split Data ---
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    except Exception as e:
        st.error(f"Error during train/test split: {e}")
        return

    # proceed to training (not displaying intermediate shapes)

    # --- 6. Train ---
    model = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
    try:
        model.fit(X_train, Y_train)
    except Exception as e:
        st.error(f"Error training model: {e}")
        return

    # model trained silently

    # return model and metadata for personal input predictions
    numeric_medians = {col: float(fd[col].median()) for col in fd.select_dtypes(include=np.number).columns}
    return model, list(X.columns), numeric_medians


def main():
    st.title("Fraud Detection Pipeline (Streamlit)")

    # Try to load CSV from relative path (same folder as app.py)
    data_path = "application_data.csv"
    df = None
    
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Failed to read {data_path}: {e}")
            return
    else:
        # Fallback: allow file upload if CSV not found
        st.info("Please upload your `application_data.csv` file to get started.")
        uploaded_file = st.file_uploader("Upload application_data.csv", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                return
        else:
            st.stop()

    model, feature_names, medians = run_pipeline(df)

    # Personal fraud detection input form on main page
    st.header('Personal Fraud Detection Analysis')
    with st.form('personal_input'):
        st.write('Enter your financial details for fraud analysis:')
        col1, col2 = st.columns(2)
        
        with col1:
            a_amt_income = st.number_input('Annual Income (AMT_INCOME_TOTAL)', value=medians.get('AMT_INCOME_TOTAL', 0.0), min_value=0.0)
            a_amt_credit = st.number_input('Credit Amount (AMT_CREDIT)', value=medians.get('AMT_CREDIT', 0.0), min_value=0.0)
            a_amt_annuity = st.number_input('Annuity (AMT_ANNUITY)', value=medians.get('AMT_ANNUITY', 0.0), min_value=0.0)
            a_days_birth = st.number_input('Days from Birth (DAYS_BIRTH)', value=medians.get('DAYS_BIRTH', 0.0))
            a_ext1 = st.number_input('External Source 1 (Credit Score)', value=medians.get('EXT_SOURCE_1', 0.0), min_value=0.0, max_value=1.0)
        
        with col2:
            a_amt_goods = st.number_input('Property Value (AMT_GOODS_PRICE)', value=medians.get('AMT_GOODS_PRICE', 0.0), min_value=0.0)
            a_days_employed = st.number_input('Days Worked (DAYS_EMPLOYED)', value=medians.get('DAYS_EMPLOYED', 0.0))
            a_cnt_fam = st.number_input('Family Members (CNT_FAM_MEMBERS)', value=medians.get('CNT_FAM_MEMBERS', 1.0), min_value=1.0)
            a_cnt_children = st.number_input('Number of Children (CNT_CHILDREN)', value=0, min_value=0)
            a_ext2 = st.number_input('External Source 2 (Credit Score)', value=medians.get('EXT_SOURCE_2', 0.0), min_value=0.0, max_value=1.0)
        
        a_ext3 = st.number_input('External Source 3 (Credit Score)', value=medians.get('EXT_SOURCE_3', 0.0), min_value=0.0, max_value=1.0)
        submit_personal = st.form_submit_button('Analyze Fraud Risk')

    if submit_personal:
        inp = {}
        inp['AMT_CREDIT'] = a_amt_credit
        inp['AMT_INCOME_TOTAL'] = a_amt_income
        inp['AMT_ANNUITY'] = a_amt_annuity
        inp['AMT_GOODS_PRICE'] = a_amt_goods
        inp['DAYS_BIRTH'] = a_days_birth
        inp['DAYS_EMPLOYED'] = a_days_employed
        inp['CNT_FAM_MEMBERS'] = a_cnt_fam
        inp['CNT_CHILDREN'] = a_cnt_children
        inp['EXT_SOURCE_1'] = a_ext1
        inp['EXT_SOURCE_2'] = a_ext2
        inp['EXT_SOURCE_3'] = a_ext3

        # compute engineered features used in training
        inp['CREDIT_INCOME_RATIO'] = inp['AMT_CREDIT'] / (inp['AMT_INCOME_TOTAL'] if inp['AMT_INCOME_TOTAL'] != 0 else 1)
        inp['ANNUITY_INCOME_RATIO'] = inp['AMT_ANNUITY'] / (inp['AMT_INCOME_TOTAL'] if inp['AMT_INCOME_TOTAL'] != 0 else 1)
        inp['ANNUITY_CREDIT_RATIO'] = inp['AMT_ANNUITY'] / (inp['AMT_CREDIT'] if inp['AMT_CREDIT'] != 0 else 1)
        inp['GOODS_PRICE_CREDIT_RATIO'] = (inp['AMT_GOODS_PRICE'] / inp['AMT_CREDIT']) if inp['AMT_CREDIT'] != 0 else 0
        inp['AGE_YEARS'] = abs(inp['DAYS_BIRTH'] / 365.25)
        inp['YEARS_EMPLOYED'] = abs(inp['DAYS_EMPLOYED'] / 365.25)
        if inp['YEARS_EMPLOYED'] == 365243:
            inp['YEARS_EMPLOYED'] = 0
        inp['DAYS_EMPLOYED_PERC'] = inp['YEARS_EMPLOYED'] / (inp['AGE_YEARS'] if inp['AGE_YEARS'] != 0 else 1)
        inp['DAYS_EMPLOYED_PERC'] = 0 if np.isnan(inp['DAYS_EMPLOYED_PERC']) else inp['DAYS_EMPLOYED_PERC']
        inp['CNT_NON_CHILDREN'] = inp['CNT_FAM_MEMBERS'] - inp['CNT_CHILDREN']
        inp['CHILDREN_RATIO'] = (inp['CNT_CHILDREN'] / inp['CNT_FAM_MEMBERS']) if inp['CNT_FAM_MEMBERS'] != 0 else 0
        inp['INCOME_PER_PERSON'] = (inp['AMT_INCOME_TOTAL'] / inp['CNT_FAM_MEMBERS']) if inp['CNT_FAM_MEMBERS'] != 0 else 0
        inp['EXT_SOURCES_PROD'] = inp['EXT_SOURCE_1'] * inp['EXT_SOURCE_2'] * inp['EXT_SOURCE_3']
        inp['DAYS_EMPLOYED_SQ'] = inp['DAYS_EMPLOYED'] ** 2

        # build DataFrame and align columns
        input_df = pd.DataFrame([inp])
        for col in feature_names:
            if col not in input_df.columns:
                # use median if available, else 0
                input_df[col] = medians.get(col, 0.0)

        input_df = input_df[feature_names]

        # predict
        try:
            prob = float(model.predict_proba(input_df)[:, 1][0])
            pred = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prob = None
            pred = None

        # show only fraud prediction result
        st.markdown("---")
        if prob is not None:
            col_left, col_right = st.columns(2)
            with col_left:
                st.metric('Fraud Risk Score', f'{round(prob * 100, 2)}%')
            with col_right:
                risk_level = 'HIGH RISK' if pred == 1 else 'LOW RISK'
                st.metric('Risk Assessment', risk_level)


if __name__ == '__main__':
    main()
