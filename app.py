import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import string
import os
import gdown
import nltk
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
st.set_page_config(
    page_title="Phân Loại Chủ Đề Văn Bản - Nhóm 6",
    page_icon=":speech_balloon:",
    layout="wide"
)

# Tải dữ liệu Stopwords từ NLTK để làm sạch văn bản
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# 2. HÀM LÀM SẠCH VĂN BẢN
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\d+', '', text)  
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]  
    return " ".join(cleaned_words)

# Ánh xạ nhãn hiển thị trực quan ngắn gọn cho trục biểu đồ
TOPIC_MAPPING = {
    1: "Society & Culture",
    2: "Science & Math",
    3: "Health",
    4: "Education & Reference",
    5: "Computers & Internet",
    6: "Sports",
    7: "Business & Finance",
    8: "Entertainment & Music",
    9: "Family & Relationships",
    10: "Politics & Government"
}

# 3. TỰ ĐỘNG TẢI TÀI NGUYÊN TỪ DRIVE KHI KHỞI CHẠY
@st.cache_resource
def load_prediction_artifacts():
    model_file = "lstm_best_model.keras"  
    tokenizer_file = "tokenizer.pkl"
    label_encoder_file = "label_encoder.pkl"
    
    # Tải trực tiếp file mô hình định dạng .keras từ Drive
    if not os.path.exists(model_file):
        with st.spinner("Đang tải file mô hình mạng LSTM từ Google Drive..."):
            drive_id_model = "1AZ42RqycaBXszQBQpyDO8sn8JeGkIrKP" 
            url = f"https://drive.google.com/uc?id={drive_id_model}"
            gdown.download(url, model_file, quiet=False)
            
    # Tải file cấu hình Tokenizer từ liên kết
    if not os.path.exists(tokenizer_file):
        with st.spinner("Đang tải bộ mã hóa Tokenizer từ Google Drive..."):
            drive_id_tok = "1MwwxulGidhRVco8KVvepIkyoXBxAYVIW"
            url = f"https://drive.google.com/uc?id={drive_id_tok}"
            gdown.download(url, tokenizer_file, quiet=False)

    # Tải file bộ mã hóa nhãn Label Encoder từ liên kết
    if not os.path.exists(label_encoder_file):
        with st.spinner("Đang tải bộ mã hóa nhãn từ Google Drive..."):
            drive_id_lbl = "1J5Skv7XXT_xhzA2PMhHRDrK3gHkXL42Z"
            url = f"https://drive.google.com/uc?id={drive_id_lbl}"
            gdown.download(url, label_encoder_file, quiet=False)

    # Nạp mô hình Keras trực tiếp từ file đơn lẻ .keras
    model = tf.keras.models.load_model(model_file)
    
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open(label_encoder_file, 'rb') as f:
        label_encoder = pickle.load(f)
        
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_prediction_artifacts()
    MAX_LENGTH = model.input_shape[1] if model.input_shape[1] is not None else 120
except Exception as e:
    st.error(f"Lỗi hệ thống khi tải hoặc nạp tài nguyên cấu hình từ Drive: {e}")
    st.stop()

# 4. THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG CHIA THEO CÁC TABS
st.title(":speech_balloon: Hệ Thống Phân Loại Chủ Đề Chuỗi Văn Bản Ngắn")
st.markdown("### Bài Tập Nhóm 2 - (Nhóm 6)")
st.markdown("---")

tab1, tab2 = st.tabs([":crystal_ball: Phân Tích Trực Quan", ":bar_chart: Đánh Giá Hiệu Năng Mô Hình"])

# TAB PHÂN TÍCH NHẬN DIỆN VĂN BẢN
with tab1:
    st.write("Mô hình sử dụng mạng học sâu **LSTM (Long Short-Term Memory)** để nhận diện 10 chủ đề chính.")
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader(":speech_balloon: Nhập nội dung văn bản câu hỏi")
        user_input = st.text_area(
            label="Nhập chuỗi văn bản (Bằng Tiếng Anh):",
            height=180,
            value="What is the best way to learn computer programming online for free?",
            placeholder="Nhập hoặc dán đoạn văn bản cần kiểm tra vào đây...",
            key="input_text"
        )
        predict_btn = st.button(":rocket: Tiến Hành Phân Loại", type="primary", width="stretch")

    with col2:
        st.subheader(":bar_chart: Kết quả dự đoán từ mô hình")
        if predict_btn:
            if user_input.strip() == "":
                st.warning("Vui lòng nhập nội dung văn bản trước khi bấm phân loại!")
            else:
                with st.spinner("Đang xử lý làm sạch văn bản và dự đoán..."):
                    cleaned = clean_text(user_input)
                    sequences = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
                    
                    predictions = model.predict(padded)[0]
                    max_idx = np.argmax(predictions)
                    
                    try:
                        raw_label = label_encoder.inverse_transform([max_idx])[0]
                        predicted_topic = TOPIC_MAPPING.get(int(raw_label), f"Chủ đề {raw_label}")
                    except:
                        predicted_topic = TOPIC_MAPPING.get(max_idx + 1, f"Chủ đề {max_idx}")
                    
                    confidence = predictions[max_idx] * 100

                    st.success(f"**Chủ đề được nhận diện:** {predicted_topic}")
                    st.metric(label="Độ tin cậy chính xác (Confidence Score)", value=f"{confidence:.2f}%")
                    
                    st.markdown("---")
                    st.write("**:bar_chart: Biểu đồ phân bổ xác suất trên cả 10 danh mục (%):**")
                    
                    chart_data = []
                    for idx, prob in enumerate(predictions):
                        topic_name = TOPIC_MAPPING.get(idx + 1, f"Chủ đề {idx+1}")
                        chart_data.append({
                            "Chủ đề": topic_name,
                            "Xác suất (%)": float(prob * 100)
                        })
                    df = pd.DataFrame(chart_data)
                    st.bar_chart(data=df, x="Chủ đề", y="Xác suất (%)")
        else:
            st.info("Mời nhập câu test ở ô bên trái, sau đó bấm nút **Tiến Hành Phân Loại**!")


# TAB ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH LSTM
with tab2:
    st.subheader(":bar_chart: Lịch sử Huấn luyện (Training & Validation History)")
    
    # Tải và hiển thị file ảnh chứa sơ đồ Accuracy và Loss từ Drive
    path_history = "lstm_training_chart.png"
    if not os.path.exists(path_history):
        with st.spinner("Đang tải sơ đồ lịch sử huấn luyện từ Drive..."):
            drive_id_history = "1XEPffCeKYu9jigj5ZO5lW3wvttl7L_K7"
            url_history = f"https://drive.google.com/uc?id={drive_id_history}"
            try:
                gdown.download(url_history, path_history, quiet=True)
            except Exception as e:
                st.error(f"Không thể tải ảnh lịch sử huấn luyện: {e}")
            
    if os.path.exists(path_history):
        st.image(path_history, caption="Đồ thị diễn biến Accuracy và Loss qua các Epoch huấn luyện", width="stretch")

    st.markdown("---")
    
    # Chia phần dưới làm 2 cột bằng nhau cho Confusion Matrix và Classification Report
    col_bottom1, col_bottom2 = st.columns(2)
    
    with col_bottom1:
        st.subheader(":jigsaw: Ma trận nhầm lẫn (Confusion Matrix)")
        path_cm = "LSTM_Confusion_Matrix.png"
        if not os.path.exists(path_cm):
            with st.spinner("Đang tải sơ đồ Confusion Matrix từ Drive..."):
                drive_id_cm = "1_yW0tyUau-zKW668Fcv43K4ivwbcNa6C"
                url_cm = f"https://drive.google.com/uc?id={drive_id_cm}"
                try:
                    gdown.download(url_cm, path_cm, quiet=True)
                except Exception as e:
                    st.error(f"Không thể tải ảnh Confusion Matrix: {e}")
                
        if os.path.exists(path_cm):
            st.image(path_cm, caption="Ma trận nhầm lẫn thực nghiệm hệ thống LSTM", width="stretch")

    with col_bottom2:
        st.subheader(":clipboard: Báo cáo phân loại (Classification Report)")
        path_report = "LSTM_Classification_Report.png"
        if not os.path.exists(path_report):
            with st.spinner("Đang tải sơ đồ Classification Report từ Drive..."):
                drive_id_report = "1D3XoL0rSAJ0HWeseuTJGg5RWpCvCvLks"
                url_report = f"https://drive.google.com/uc?id={drive_id_report}"
                try:
                    gdown.download(url_report, path_report, quiet=True)
                except Exception as e:
                    st.error(f"Không thể tải ảnh Classification Report: {e}")
                
        if os.path.exists(path_report):
            st.image(path_report, caption="Bảng chi tiết chỉ số Precision, Recall, F1-Score của mô hình LSTM", width="stretch")

    # SO SÁNH CÁC MÔ HÌNH
    st.markdown("---")
    st.subheader(":bar_chart: So Sánh Hiệu Năng Giữa Các Mô Hình Thực Nghiệm")
    st.write("Biểu đồ so sánh trực quan các chỉ số đo lường hiệu năng tổng quan giữa kiến trúc mạng LSTM với các kiến trúc đối sánh khác trong bài toán.")
    
    path_compare = "models_comparison_chart.png"
    if not os.path.exists(path_compare):
        with st.spinner("Đang tải sơ đồ đối sánh mô hình từ Drive..."):
            drive_id_compare = "18AeOQ5K21pXT5x7r8b_xojdiZvKl3bM3" 
            url_compare = f"https://drive.google.com/uc?id={drive_id_compare}"
            try:
                gdown.download(url_compare, path_compare, quiet=True)
            except Exception as e:
                st.error(f"Không thể tải sơ đồ đối sánh mô hình: {e}")
                
    if os.path.exists(path_compare):
        st.image(path_compare, caption="Biểu đồ cột so sánh Accuracy, Precision, Recall và F1-score của các mô hình", width="stretch")
