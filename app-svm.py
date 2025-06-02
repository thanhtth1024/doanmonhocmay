import streamlit as st
import joblib
import re
import pandas as pd
import html
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from datetime import datetime
import os
import uuid
import streamlit.components.v1 as components  # Dùng để chạy JavaScript tự refresh
import numpy as np  # Thêm thư viện numpy để xử lý với SVM

# Hàm tự làm mới trang bằng JavaScript
def auto_refresh():
    components.html("<script>window.location.reload();</script>", height=0)

# Đường dẫn file lưu lịch sử (history)
HISTORY_FILE = "./email_history.csv"

st.set_page_config(page_title="HỆ THỐNG KIỂM TRA EMAIL SPAM", page_icon="✉️", layout="wide")

# --------------------------
# Load lịch sử kiểm tra từ file nếu có, lưu vào session_state
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict("records")
        except Exception as e:
            st.session_state.history = []
    else:
        st.session_state.history = []

# Khởi tạo tập hợp lưu email đã lưu (để tránh duplicate)
if "saved_email_texts" not in st.session_state:
    st.session_state.saved_email_texts = set()

# --------------------------
# Hàm tải model và vectorizer
@st.cache_resource(ttl=3600)
def load_model():
    learn_inf = joblib.load('./checkpoints/spam_detection_model.pkl')
    vectorizer = joblib.load('./checkpoints/tfidf_vectorizer.pkl')
    return learn_inf, vectorizer

# --------------------------
# Hàm tải danh sách từ khóa spam từ dataset
@st.cache_resource
def load_spam_keywords():
    data = pd.read_csv('./spam_ham_dataset.csv')
    spam_emails = data[data['label'] == 'spam']['text']
    spam_keywords = set()
    for email in spam_emails:
        spam_keywords.update(email.lower().split())
    spam_keywords = [word for word in spam_keywords if len(word) > 3]
    return spam_keywords

# --------------------------
# Hàm lưu kết quả vào lịch sử (session_state) và cập nhật file CSV
def save_to_history(email_text, is_spam, spam_probability, word_count, spam_count):
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'email_text': email_text,
        'is_spam': is_spam,
        'spam_probability': spam_probability,
        'word_count': word_count,
        'spam_count': spam_count
    }
    st.session_state.history.append(entry)
    # Lưu lại toàn bộ lịch sử vào file CSV
    df_history = pd.DataFrame(st.session_state.history)
    df_history.to_csv(HISTORY_FILE, index=False)

# --------------------------
# Hàm phân loại email sử dụng SVM
def classify_email(model, vectorizer, email):
    # Chuyển đổi email thành vector đặc trưng
    email_vec = vectorizer.transform([email])
    
    # Dự đoán nhãn (0: ham, 1: spam)
    prediction = model.predict(email_vec)[0]
    
    # Tính xác suất dựa trên khoảng cách đến siêu phẳng (decision function)
    # SVM không có hàm predict_proba() mặc định, nên ta sẽ chuyển đổi decision_function thành xác suất
    decision_score = model.decision_function(email_vec)[0]
    
    # Chuyển đổi decision_function thành xác suất bằng hàm sigmoid
    spam_probability = 1 / (1 + np.exp(-decision_score))
    
    return prediction, spam_probability


def highlight_keywords(text, keywords):
    text = html.escape(text)
    text = text.replace('\n', '<br>')
    words = text.split(' ')
    highlighted_words = []
    for word in words:
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w\s]', '', word_lower)
        if word_clean in [k.lower() for k in keywords]:
            highlighted_words.append(f'<mark style="background-color: #FF9999;">{word}</mark>')
        else:
            highlighted_words.append(word)
    return ' '.join(highlighted_words)

# --------------------------
# Hàm thống kê nội dung email
def get_email_statistics(email_text, spam_keywords):
    words = email_text.split()
    word_count = len(words)
    spam_word_counts = {}
    total_spam_words = 0
    for word in words:
        word_clean = re.sub(r'[^\w\s]', '', word.lower())
        if word_clean in [k.lower() for k in spam_keywords]:
            spam_word_counts[word_clean] = spam_word_counts.get(word_clean, 0) + 1
            total_spam_words += 1
    return word_count, total_spam_words, spam_word_counts

# --------------------------
# Hàm tạo biểu đồ phân bố từ khóa trong email
def create_distribution_chart(word_count, spam_count):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Từ thường', 'Từ khóa spam'],
        y=[word_count - spam_count, spam_count],
        marker_color=['#00CC96', '#EF553B']
    ))
    fig.update_layout(
        title='Phân bố từ khóa spam trong email',
        yaxis_title='Số lượng từ',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    return fig

# --------------------------
# Hàm tạo biểu đồ xu hướng spam từ lịch sử
def create_trend_chart(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['spam_probability'],
        mode='lines+markers',
        name='Xác suất spam',
        line=dict(color='#EF553B')
    ))
    fig.update_layout(
        title='Xu hướng phát hiện spam theo thời gian',
        xaxis_title='Thời gian',
        yaxis_title='Xác suất spam',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    return fig

# --------------------------
# Hàm hiển thị giao diện phân tích cho 1 email
def display_email_analysis(email_text, source="", detailed=False):
    model, vectorizer = load_model()
    spam_keywords = load_spam_keywords()
    prediction, spam_probability = classify_email(model, vectorizer, email_text)
    if source:
        st.markdown(f"**Nguồn:** {source}")
    if prediction == 0:
        st.success("✅ Email này không phải là Spam")
        word_count, spam_count, _ = get_email_statistics(email_text, spam_keywords)
        return prediction, spam_probability, word_count, spam_count
    else:
        word_count, spam_count, spam_words = get_email_statistics(email_text, spam_keywords)
        if not detailed:
            st.error("⚠️ Email này được xác định là Spam!")
            # st.markdown(f"**Xác suất spam: {spam_probability:.1%}**")
            st.subheader("Gợi ý hành động")
            st.error("Email có độ nguy hiểm cao! Nên xóa ngay lập tức và chặn người gửi.")
            st.markdown("Để xem phân tích chi tiết, vui lòng chuyển sang tab **Phân tích chi tiết**.")
        else:
            highlighted_text = highlight_keywords(email_text, spam_keywords)
            st.markdown(f'<div class="email-container">{highlighted_text}</div>', unsafe_allow_html=True)
            st.error("⚠️ Email này được xác định là Spam!")
            # st.progress(spam_probability, text=f"Xác suất spam: {spam_probability:.1%}")
            st.progress(spam_probability)
            st.subheader("Gợi ý hành động")
            if spam_probability > 0.8:
                st.error("Email có độ nguy hiểm cao! Nên xóa ngay lập tức và chặn người gửi.")
            else:
                st.warning("Cân nhắc đánh dấu email này là spam hoặc xóa đi.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tổng số từ", word_count)
                st.metric("Số từ khóa spam", spam_count)
            with col2:
                fig = create_distribution_chart(word_count, spam_count)
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{hash(email_text)}_{uuid.uuid4()}")
            if spam_count > 0:
                st.subheader("Chi tiết từ khóa spam")
                spam_df = pd.DataFrame(list(spam_words.items()), columns=['Từ khóa', 'Số lần xuất hiện'])
                st.dataframe(spam_df.sort_values('Số lần xuất hiện', ascending=False))
        return prediction, spam_probability, word_count, spam_count


def main():
    
    st.title("HỆ THỐNG KIỂM TRA EMAIL SPAM")
    
    st.markdown("""
        <style>
        .email-container {
            background-color: #0e1117;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #464c55;
            margin: 10px 0;
            color: #fafafa;
        }
        mark {
            padding: 2px 4px;
            border-radius: 3px;
            background-color: #FF9999;
            color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---- Tab cấu hình ----
    tabs = st.tabs(["Kiểm tra Email thủ công", "Tải lên danh sách Email", "Phân tích chi tiết", "Lịch sử"])
    spam_keywords = load_spam_keywords()

    # ----------------------
    # Tab 1: Kiểm tra Email thủ công
    with tabs[0]:
        st.subheader("Kiểm tra Email thủ công")
        # Lấy query parameter 'email' nếu có từ extension
        params = st.query_params
        email_param = params.get("email", [""])[0]
        user_input = st.text_area('Nhập nội dung email:', value=email_param,
                                  placeholder='Ví dụ: Chúc mừng!! Bạn đã trúng thưởng Rs. 100000. Nhấn vào đây để nhận ngay...')
        if st.button("Kiểm tra Spam", key="manual"):
            if user_input.strip():
                prediction, spam_probability, word_count, spam_count = display_email_analysis(user_input)
                save_to_history(user_input, prediction == 1, spam_probability, word_count, spam_count)
                st.session_state.selected_manual_email = user_input
                # Đánh dấu email thủ công đã được lưu để tránh duplicate
                st.session_state.saved_email_texts.add(user_input)
            else:
                st.warning("Vui lòng nhập nội dung email để kiểm tra!")

    # ----------------------
    # Tab 2: Tải lên danh sách Email
    with tabs[1]:
        st.subheader("Tải lên danh sách Email")
        st.info("Tải file CSV có chứa cột 'email_text'. Nếu không tải lên, hệ thống sẽ dùng file mặc định (nếu có).")
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], key="csv")
        if uploaded_file is not None:
            try:
                df_emails = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error("Không thể đọc file CSV.")
                df_emails = pd.DataFrame(columns=['email_text'])
        else:
            try:
                df_emails = pd.read_csv('./email_inbox.csv')
            except Exception as e:
                st.error("Không tìm thấy file CSV mặc định. Vui lòng tải lên file của bạn.")
                df_emails = pd.DataFrame(columns=['email_text'])
                
        if df_emails.empty:
            st.info("Danh sách email trống.")
        else:
            if 'email_text' not in df_emails.columns:
                st.error("File CSV phải có cột 'email_text'.")
            else:
                model, vectorizer = load_model()
                results = []
                for idx, row in df_emails.iterrows():
                    email_text = row['email_text']
                    timestamp = row['timestamp'] if 'timestamp' in row and pd.notna(row['timestamp']) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    prediction, spam_probability = classify_email(model, vectorizer, email_text)
                    results.append({
                        'index': idx,
                        'timestamp': timestamp,
                        'email_text': email_text,
                        'is_spam': prediction == 1,
                        'spam_probability': spam_probability
                    })
                df_results = pd.DataFrame(results)
                df_results['snippet'] = df_results['email_text'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
                df_results['Xác suất spam'] = df_results['spam_probability'].apply(lambda x: f"{x:.1%}")
                df_results['Kết quả'] = df_results['is_spam'].map({True: '❌ Spam', False: '✅ An toàn'})
                
                st.subheader("Kết quả quét danh sách Email")
                st.dataframe(df_results[['index', 'timestamp', 'snippet', 'Kết quả']].sort_values('timestamp', ascending=False),
                    use_container_width=True)
                st.session_state.csv_results = df_results

    # ----------------------
    # Tab 3: Phân tích chi tiết cho nhiều email
    with tabs[2]:
        st.subheader("Phân tích chi tiết Email")
        # Phân tích email kiểm tra thủ công 
        if "selected_manual_email" in st.session_state:
            st.markdown("### Email kiểm tra thủ công")
            with st.container():
                email_text = st.session_state.selected_manual_email
                if email_text not in st.session_state.saved_email_texts:
                    prediction, spam_probability, word_count, spam_count = display_email_analysis(email_text, source="Kiểm tra thủ công", detailed=True)
                    save_to_history(email_text, prediction == 1, spam_probability, word_count, spam_count)
                    st.session_state.saved_email_texts.add(email_text)
                else:
                    display_email_analysis(email_text, source="Kiểm tra thủ công", detailed=True)
        else:
            st.info("Chưa có email kiểm tra thủ công nào.")

        # Phân tích email từ file CSV 
        if "csv_results" in st.session_state:
            st.markdown("### Email từ danh sách CSV")
            model, vectorizer = load_model()
            for _, row in st.session_state.csv_results.iterrows():
                with st.expander(f"Email {row['index']}: {row['snippet']}"):
                    email_text = row['email_text']
                    if email_text not in st.session_state.saved_email_texts:
                        prediction, spam_probability, word_count, spam_count = display_email_analysis(email_text, source="Danh sách CSV", detailed=True)
                        save_to_history(email_text, prediction == 1, spam_probability, word_count, spam_count)
                        st.session_state.saved_email_texts.add(email_text)
                    else:
                        display_email_analysis(email_text, source="Danh sách CSV", detailed=True)
        else:
            st.info("Chưa có kết quả từ file CSV.")

    # ----------------------
    # Tab 4: Lịch sử kiểm tra email
    with tabs[3]:
        st.subheader("Lịch sử kiểm tra email")
        history = st.session_state.get('history', [])
        if not history:
            st.info("Chưa có email nào được kiểm tra.")
        else:
            st.subheader("Xu hướng phát hiện spam")
            trend_chart = create_trend_chart(history)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True, key=f"trend_chart_{uuid.uuid4()}")
            
            total_emails = len(history)
            spam_emails = sum(1 for entry in history if entry['is_spam'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số email đã kiểm tra", total_emails)
            with col2:
                st.metric("Số email spam", spam_emails)
            with col3:
                st.metric("Tỷ lệ spam", f"{(spam_emails/total_emails*100):.1f}%" if total_emails > 0 else "0%")
            
            st.subheader("Chi tiết lịch sử kiểm tra")
            history_df = pd.DataFrame(history)
            history_df['Thời gian'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            history_df['Kết quả'] = history_df['is_spam'].map({True: '❌ Spam', False: '✅ An toàn'})
            history_df['Xác suất spam'] = history_df['spam_probability'].map('{:.1%}'.format)
            display_df = history_df[['Thời gian', 'email_text', 'Kết quả', 'Xác suất spam']]
            display_df.columns = ['Thời gian', 'Nội dung email', 'Kết quả', 'Xác suất spam']
            st.dataframe(display_df[['Thời gian', 'Nội dung email', 'Kết quả']].sort_values('Thời gian', ascending=False), 
                use_container_width=True)
            
            # CHỨC NĂNG: Chọn email từ lịch sử để phân tích lại
            st.markdown("## Phân tích lại email từ lịch sử")
            options = { f"{i+1}. {entry['timestamp']} - {entry['email_text'][:50]}": entry['email_text'] 
                        for i, entry in enumerate(history) }
            selected_option = st.selectbox("Chọn email từ lịch sử", list(options.keys()))
            email_to_reanalyze = options[selected_option]
            st.markdown("### Kết quả phân tích lại:")
            display_email_analysis(email_to_reanalyze, source="Từ Lịch sử", detailed=True)
            
            if st.button("Xóa lịch sử"):
                st.session_state.history = []
                st.session_state.saved_email_texts = set()
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.success("Đã xóa toàn bộ lịch sử!")
                auto_refresh()

if __name__ == "__main__":
    main()
