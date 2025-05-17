import streamlit as st
import pandas as pd
import pdfplumber
import io
from PIL import Image
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import google.generativeai as genai  

GEMINI_API_KEY = "GEMINI API KEY HERE"  
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

deposit_keywords = ["IMPS-MOB/Fund Trf", "CREDIT", "APB-INW", "NEFT", "RTGS", "UPI", "Transfer from", "Deposit", "Deposit to"]
withdrawal_keywords = ["ATM-NFS/CASH WITHDRAWAL", "POS-VISA", "IMPS-RIB/Fund Trf", "DEBIT", "NACH/TP ACH", "BILLPAY/RECH", "Charge:", "UPI", "Transfer to", "Withdrawal", "Payment by"]
salary_keywords = ["salary", "Payroll", "PAYMENT", "Salary Credited"] 
rent_keywords = ["RENT", "Rent Payment", "LEASE", "Rental Payment", "House Rent"] 
utility_keywords = ["ELECTRICITY", "UTILITY", "WATER BILL", "GAS BILL", "VODAFONE PRE", "AIRTEL PREPAID", "VIDEOCON DTH", "TATASKY DTH", "PHONEPE RECHARGE", "Electricity Bill", "Water Bill Payment", "Gas Payment", "Utilities Payment"] 
loan_keywords = ["Bajaj Finanac", "loan", "EMI", "installment", "FINANAC", "Loan EMI", "Loan Payment", "Installment Payment"] 

def extract_text_from_pdf_pure_python_pages(pdf_file, pages_per_chunk=5):
    page_texts = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i in range(0, len(pdf.pages), pages_per_chunk):
                chunk_text = ""
                for page_num in range(i, min(i + pages_per_chunk, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    chunk_text += page.extract_text() + "\n\n" 
                page_texts.append(chunk_text)
        return page_texts 
    except Exception as e:
        st.error(f"Error during PDF processing (pdfplumber): {e}")
        return None

def parse_statement_data_with_ai(text_chunks):
    all_transactions = [] 

    if not text_chunks:
        return pd.DataFrame() 

    for chunk_index, text_chunk in enumerate(text_chunks):
        prompt = f"""
        Analyze the following chunk of bank statement text (Chunk {chunk_index + 1}) and extract transaction details into a structured format.
        Identify columns for 'Transaction Date', 'Value Date', 'Particulars' (transaction description), 'Debit Amount', 'Credit Amount', and 'Balance'.
        For 'Transaction Date' and 'Value Date', ALWAYS output the date with the **full year (YYYY)**, in the format **DD-MMM-YYYY** (e.g., 15-Oct-2023, 02-Jan-2024).  Do not output dates without the year. If the year is not explicitly mentioned in the text, infer the year from the context of the bank statement if possible, or use the current year if inference is not reliable.
        If 'Value Date' is not explicitly present, use 'Transaction Date' as 'Value Date' and ensure it includes the year.
        If 'Debit Amount' and 'Credit Amount' are not separate, infer them from 'Particulars' if possible (e.g., keywords like 'Debit', 'Credit', 'Withdrawal', 'Deposit'). If a single 'Amount' column exists, determine if it's debit or credit based on context or keywords in 'Particulars'.
        If 'Cheque No.' is present, extract it; otherwise, leave it blank.
        Return the extracted data as a valid JSON array of JSON objects, where each object represents a transaction with the keys: 'Transaction Date', 'Value Date', 'Particulars', 'Cheque No', 'Debit', 'Credit', 'Balance'. Ensure the output is properly formatted JSON and parsable by a JSON parser. Do not include any markdown formatting like ```json or ``` around the JSON output. Just return the raw JSON.

        Example Output Format:
        [
            {{"Transaction Date": "DD-MMM-YYYY", "Value Date": "DD-MMM-YYYY", "Particulars": "Description", "Cheque No": "Number or null", "Debit": "Amount", "Credit": "Amount", "Balance": "Amount"}},
            {{"Transaction Date": "...", "Value Date": "...", "Particulars": "...", "Cheque No": "...", "Debit": "...", "Credit": "...", "Balance": "..."}},
            ...
        ]

        Bank Statement Text Chunk:
        ```{text_chunk}```
        """

        try:
            response = model.generate_content(prompt)
            ai_output = response.text

            print(f"--- AI Output (Raw) - Chunk {chunk_index + 1} ---") 
            print(ai_output)
            print(f"--- End AI Output - Chunk {chunk_index + 1} ---")

            ai_output = ai_output.removeprefix("```json").removesuffix("```").strip()
            ai_output = ai_output.removeprefix("```").removesuffix("```").strip() 

            try:
                import json
                transactions_list_chunk = json.loads(ai_output) 

                if not isinstance(transactions_list_chunk, list): 
                    st.error(f"AI output for Chunk {chunk_index + 1} was not parsed as a list of transactions. Review AI output format.")
                    continue 

                all_transactions.extend(transactions_list_chunk) 

            except json.JSONDecodeError as e: 
                st.error(f"Error decoding AI output as JSON for Chunk {chunk_index + 1}: {e}. Please check the AI output format.")
                st.error(f"Raw AI Output (for debugging - Chunk {chunk_index + 1} - check format):")
                st.error(ai_output) 
                continue 

        except Exception as e:
            st.error(f"Error during AI-powered parsing for Chunk {chunk_index + 1}: {e}")
            continue 

    if not all_transactions: 
        st.warning("No transactions could be extracted from any of the text chunks. AI parsing might have failed for all chunks or returned empty results.")
        return pd.DataFrame() 

    df = pd.DataFrame(all_transactions) 

    df.rename(columns={
        'Transaction Date': 'Transaction Date',
        'TransactionDate': 'Transaction Date',
        'Value Date': 'Value Date',
        'ValueDate': 'Value Date',
        'Particulars': 'Particulars',
        'Description': 'Particulars', 
        'Cheque No': 'Cheque No.', 
        'ChequeNo': 'Cheque No.',
        'Cheque No.': 'Cheque No.', 
        'Debit Amount': 'Debit',
        'DebitAmount': 'Debit',
        'Debit': 'Debit',
        'Credit Amount': 'Credit',
        'CreditAmount': 'Credit',
        'Credit': 'Credit',
        'Balance': 'Balance',
        'Balance Amount': 'Balance',
        'BalanceAmount': 'Balance',
        'Amount': 'Amount' 
    }, inplace=True)

    for col in ['Transaction Date', 'Value Date', 'Particulars', 'Cheque No', 'Debit', 'Credit', 'Balance']:
        if col not in df.columns:
            df[col] = None 

    for col in ['Debit', 'Credit', 'Balance']:
        if col in df.columns:
            df[col] = df[col].fillna('0.00').astype(str).str.replace(r'[^\d\.\-]+', '', regex=True).replace('', '0.00') 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    return df

def train_transaction_categorizer(df, retrain=False):
    model_filename = "transaction_category_model.joblib"

    if not retrain and os.path.exists(model_filename):
        st.info("Loading pre-trained ML model...")
        return joblib.load(model_filename)

    st.info("Training ML transaction categorizer...")

    labeled_df = categorize_transactions_keyword_based(df.copy())
    labeled_df = labeled_df[labeled_df['Category'] != 'Uncategorized'].copy()

    st.write("--- transactions_df before keyword categorization ---")
    st.dataframe(df) 
    st.write("--- labeled_df after keyword categorization ---")
    st.dataframe(labeled_df) 

    if labeled_df.empty:
        st.warning("Insufficient labeled data to train the ML model. Please ensure your statements have recognizable transactions OR **expand keyword dictionaries to cover more transaction types.**") 
        return None

    st.write("Category Value Counts before train_test_split:")
    st.write(labeled_df['Category'].value_counts())

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(labeled_df['Particulars'])
    y = labeled_df['Category']

    min_class_size = y.value_counts().min()
    if min_class_size >= 2:
        st.info("Stratifying train_test_split based on category.")
        stratify_param = y 
    else:
        st.warning(f"Not stratifying train_test_split because the smallest category has only {min_class_size} sample(s) (< 2). This might lead to imbalanced train/test sets for some categories.")
        stratify_param = None 

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    except ValueError as e:
        st.error(f"Error during train_test_split: {e}")
        st.error("This is likely due to very small category sizes after keyword-based labeling. Consider reviewing your keyword dictionaries or providing more data with better representation across categories.")
        return None 

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model training accuracy (on limited test set): {accuracy:.2f}")

    model_package = {'classifier': classifier, 'vectorizer': vectorizer}
    joblib.dump(model_package, model_filename)
    st.success(f"ML model trained and saved to {model_filename}")
    return model_package

def categorize_transactions_ml_based(df, model_package):
    if df.empty or model_package is None:
        return df

    vectorizer = model_package['vectorizer']
    classifier = model_package['classifier']

    X_new = vectorizer.transform(df['Particulars'])
    predicted_categories = classifier.predict(X_new)
    df['Category'] = predicted_categories

    return df

def analyze_financial_health(df):
    if df.empty:
        return {"monthly_summary": pd.DataFrame(), "regular_bills": pd.DataFrame(), "loan_payments": pd.DataFrame(), "balance_trend": pd.DataFrame(), "loan_recommendation": "Insufficient data"}

    for col in ['Debit', 'Credit', 'Balance']:
        if col in df.columns:

            df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]+', '', regex=True) 

            df[col] = df[col].replace('', '0.00')

            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    df['Value Date'] = pd.to_datetime(df['Value Date'], errors='coerce')
    df.dropna(subset=['Value Date'], inplace=True) 

    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)

    df['Month'] = df['Value Date'].dt.to_period('M')

    monthly_summary = df.groupby('Month').agg(
        TotalDeposits=('Credit', 'sum'),
        TotalWithdrawals=('Debit', 'sum')
    ).reset_index()

    monthly_summary['NetFlow'] = monthly_summary['TotalDeposits'] - monthly_summary['TotalWithdrawals'] 

    regular_bills = df[df['Category'].isin(['Rent', 'Utilities'])].groupby('Category').agg(
        TotalAmount=('Debit', 'sum'),
        Transactions=('Debit', 'count')
    ).reset_index()

    loan_payments = df[df['Category'] == 'Loan Repayment'].groupby('Category').agg(
        TotalAmount=('Debit', 'sum'),
        Transactions=('Debit', 'count')
    ).reset_index()

    balance_trend = df[['Value Date', 'Balance']].sort_values(by='Value Date').reset_index(drop=True)

    total_deposits = monthly_summary['TotalDeposits'].sum()
    total_withdrawals = monthly_summary['TotalWithdrawals'].sum()
    net_flow_overall = total_deposits - total_withdrawals
    loan_repayment_amount = loan_payments['TotalAmount'].sum() if not loan_payments.empty else 0

    if net_flow_overall > 0 and loan_repayment_amount < total_deposits * 0.2:
        loan_recommendation = "Strong Candidate: Positive cash flow and manageable debt."
    elif net_flow_overall > 0 and loan_repayment_amount >= total_deposits * 0.2 and loan_repayment_amount < total_deposits * 0.5:
        loan_recommendation = "Moderate Risk: Positive cash flow but significant debt. Further review needed."
    elif net_flow_overall <= 0 and loan_repayment_amount < total_deposits * 0.2:
         loan_recommendation = "Moderate Risk: Negative cash flow, but low debt. Needs improvement in income or spending."
    elif net_flow_overall <= 0:
        loan_recommendation = "High Risk: Negative cash flow and potentially high debt. Loan unlikely without significant improvement."
    else:
        loan_recommendation = "Neutral: Financial health is unclear, requires more data."

    return {
        "monthly_summary": monthly_summary,
        "regular_bills": regular_bills,
        "loan_payments": loan_payments,
        "balance_trend": balance_trend,
        "loan_recommendation": loan_recommendation
    }

def categorize_transactions_keyword_based(df):
    if df.empty:
        return df

    df['Category'] = 'Uncategorized'

    for index, row in df.iterrows():
        particulars = str(row['Particulars']).lower()

        if any(keyword.lower() in particulars for keyword in salary_keywords):
            df.loc[index, 'Category'] = 'Salary'
        elif any(keyword.lower() in particulars for keyword in rent_keywords):
            df.loc[index, 'Category'] = 'Rent'
        elif any(keyword.lower() in particulars for keyword in utility_keywords):
            df.loc[index, 'Category'] = 'Utilities'
        elif any(keyword.lower() in particulars for keyword in loan_keywords):
            df.loc[index, 'Category'] = 'Loan Repayment'
        elif any(keyword.lower() in particulars for keyword in deposit_keywords):
            df.loc[index, 'Category'] = 'Deposit'
        elif any(keyword.lower() in particulars for keyword in withdrawal_keywords):
            df.loc[index, 'Category'] = 'Withdrawal'
    return df

st.title("Bank Statement Analyzer (ML-Powered)")
st.write("Upload a bank statement PDF to analyze financial health using Machine Learning and get loan recommendations.")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")
retrain_model = st.checkbox("Retrain ML Model (check to retrain, uncheck to use saved model)")
pages_per_chunk_input = st.number_input("Pages per AI Chunk", min_value=1, max_value=10, value=5, step=1, help="Process PDF in chunks to avoid AI token limits. Adjust if needed.") 

if uploaded_file is not None:
    st.header("1. Raw Extracted Text") 
    file_path = "temp_statement.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    extracted_text_chunks = extract_text_from_pdf_pure_python_pages(file_path, pages_per_chunk=pages_per_chunk_input) 
    if extracted_text_chunks:
        combined_extracted_text = "\n\n----PAGE BREAK----\n\n".join(extracted_text_chunks) 
        st.text_area("Extracted Text from PDF (Chunked):", value=combined_extracted_text or "Text extraction failed.", height=300) 
    else:
        st.error("Text extraction from PDF failed using pure Python methods. The PDF might be image-based or complex.")
        os.remove(file_path) 
        st.stop()

    if extracted_text_chunks: 
        st.header("2. Extracted Transactions (AI-Powered)")
        transactions_df = parse_statement_data_with_ai(extracted_text_chunks) 

        if not transactions_df.empty:
            st.dataframe(transactions_df)

            st.header("3. Financial Insights (ML Categorization)")
            model_package = train_transaction_categorizer(transactions_df.copy(), retrain=retrain_model)
            if model_package:
                categorized_df = categorize_transactions_ml_based(transactions_df.copy(), model_package)

                financial_analysis = analyze_financial_health(categorized_df)

                st.subheader("Monthly Summary")
                st.dataframe(financial_analysis["monthly_summary"])

                
                # st.subheader("Regular Bills (Rent, Utilities)")
                # if not financial_analysis["regular_bills"].empty:
                #     st.dataframe(financial_analysis["regular_bills"])
                # else:
                #     st.write("No regular bills identified.")

                # st.subheader("Loan Repayments")
                # if not financial_analysis["loan_payments"].empty:
                #     st.dataframe(financial_analysis["loan_payments"])
                # else:
                #     st.write("No loan repayments identified.")

                

                st.subheader("Balance Trend (All Data Points)") 
                if not financial_analysis["balance_trend"].empty:

                    st.line_chart(data=financial_analysis["balance_trend"].set_index('Value Date'), y='Balance')
                else:
                    st.write("Balance trend not available due to data issues.")

                st.header("4. Loan Recommendation")
                st.write(f"**Loan Recommendation:** {financial_analysis['loan_recommendation']}")
            else:
                st.error("ML model could not be trained or loaded. Categorization and further insights are unavailable.")

        else:
            st.warning("Could not parse transaction data from the statement using AI. The format might be too complex or AI failed to understand the structure. Please review the extracted text and consider manual adjustments if needed.")
    else:
        st.error("Text extraction from PDF failed. Cannot proceed with AI parsing.")

    os.remove(file_path)

if __name__ == '__main__':

    @st.cache_data
    def categorize_transactions_keyword_based(df):
        return categorize_transactions_keyword_based_impl(df)

    def categorize_transactions_keyword_based_impl(df):
        if df.empty:
            return df

        df['Category'] = 'Uncategorized'

        for index, row in df.iterrows():
            particulars = str(row['Particulars']).lower()

            if any(keyword.lower() in particulars for keyword in salary_keywords):
                df.loc[index, 'Category'] = 'Salary'
            elif any(keyword.lower() in particulars for keyword in rent_keywords):
                df.loc[index, 'Category'] = 'Rent'
            elif any(keyword.lower() in particulars for keyword in utility_keywords):
                df.loc[index, 'Category'] = 'Utilities'
            elif any(keyword.lower() in particulars for keyword in loan_keywords):
                df.loc[index, 'Category'] = 'Loan Repayment'
            elif any(keyword.lower() in particulars for keyword in deposit_keywords):
                df.loc[index, 'Category'] = 'Deposit'
            elif any(keyword.lower() in particulars for keyword in withdrawal_keywords):
                df.loc[index, 'Category'] = 'Withdrawal'
        return df