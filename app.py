import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import string
import os
import gdown
import nltk
import zipfile  # Thêm thư viện giải nén cấu trúc SavedModel
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Phân Loại Chủ Đề Văn Bản - Nhóm 6",
    page_icon="📝",
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

# ----------------------------------------------------------------
# 2. HÀM LÀM SẠCH VĂN BẢN (ĐỒNG BỘ 100% VỚI FILE TRAIN NOTEBOOK)
# ----------------------------------------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Xóa URL
    text = re.sub(r'<.*?>', '', text)  # Xóa thẻ HTML
    text = text.translate(str.maketrans('', '', string.punctuation))  # Xóa dấu câu
    text = re.sub(r'\d+', '', text)  # Xóa chữ số
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]  # Xóa Stopwords
    return " ".join(cleaned_words)

# Ánh xạ nhãn hiển thị cho 10 danh mục của Yahoo Answers
TOPIC_MAPPING = {
    1: "Society & Culture (Xã hội & Văn hóa)",
    2: "Science & Mathematics (Khoa học & Toán học)",
    3: "Health (Sức khỏe)",
    4: "Education & Reference (Giáo dục & Tra cứu)",
    5: "Computers & Internet (Máy tính & Internet)",
    6: "Sports (Thể thao)",
    7: "Business & Finance (Kinh doanh & Tài chính)",
    8: "Entertainment & Music (Giải trí & Âm nhạc)",
    9: "Family & Relationships (Gia đình & Mối quan hệ)",
    10: "Politics & Government (Chính trị & Chính phủ)"
}

# ----------------------------------------------------------------
# 3. TỰ ĐỘNG TẢI FILE TỪ DRIVE KHI KHỞI CHẠY (HỖ TRỢ ĐỊNH DẠNG SAVEDMODEL ZIP)
# ----------------------------------------------------------------
@st.cache_resource
def load_prediction_artifacts():
    model_zip = "lstm_yahoo_model.zip"
    model_dir = "lstm_yahoo_model"  # Thư mục chứa cấu trúc saved_model.pb sau giải nén
    tokenizer_file = "tokenizer.pkl"
    label_encoder_file = "label_encoder.pkl"
    
    # 1. Xử lý tải và giải nén thư mục mô hình TensorFlow SavedModel
    if not os.path.exists(model_dir):
        if not os.path.exists(model_zip):
            with st.spinner("Đang tải file nén mô hình từ Google Drive..."):
                # --- ÔNG NHỚ THAY ID FILE ZIP TRÊN GOOGLE DRIVE CỦA ÔNG VÀO ĐÂY NHÉ ---
                drive_id_model = "1AZ42RqycaBXszQBQpyDO8sn8JeGkIrKP" 
                url = f"https://drive.google.com/uc?id={drive_id_model}"
                gdown.download(url, model_zip, quiet=False)
        
        # Tiến hành giải nén file zip ra thư mục trên server Streamlit
        with st.spinner("Đang giải nén cấu trúc SavedModel..."):
            with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            # Giải phóng dung lượng ổ đĩa của server sau khi giải nén xong
            if os.path.exists(model_zip):
                os.remove(model_zip)
            
    # 2. Tải file cấu hình Tokenizer từ liên kết có sẵn
    if not os.path.exists(tokenizer_file):
        with st.spinner("Đang tải bộ mã hóa Tokenizer từ Google Drive..."):
            drive_id_tok = "1MwwxulGidhRVco8KVvepIkyoXBxAYVIW"
            url = f"https://drive.google.com/uc?id={drive_id_tok}"
            gdown.download(url, tokenizer_file, quiet=False)

    # 3. Tải file bộ mã hóa nhãn Label Encoder từ liên kết có sẵn
    if not os.path.exists(label_encoder_file):
        with st.spinner("Đang tải bộ mã hóa nhãn từ Google Drive..."):
            drive_id_lbl = "1J5Skv7XXT_xhzA2PMhHRDrK3gHkXL42Z"
            url = f"https://drive.google.com/uc?id={drive_id_lbl}"
            gdown.download(url, label_encoder_file, quiet=False)

    # Nạp mô hình Keras trực tiếp từ thư mục SavedModel đã giải nén
    model = tf.keras.models.load_model(model_dir)
    
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open(label_encoder_file, 'rb') as f:
        label_encoder = pickle.load(f)
        
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_prediction_artifacts()
    # Tự động lấy độ dài maxlen từ cấu hình lớp đầu vào của mạng LSTM
    MAX_LENGTH = model.input_shape[1] if model.input_shape[1] is not None else 120
except Exception as e:
    st.error(f"Lỗi hệ thống khi tải hoặc nạp tài nguyên cấu hình từ Drive: {e}")
    st.stop()

# ----------------------------------------------------------------
# 4. THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG (UI/UX)
# ----------------------------------------------------------------
st.title("📝 Hệ Thống Phân Loại Chủ Đề Chuỗi Văn Bản Ngắn")
st.markdown("### Bài Tập Nhóm 2 - Lớp Thực Hành Deep Learning (Nhóm 6)")
st.write("Mô hình sử dụng mạng học sâu **LSTM (Long Short-Term Memory)** để nhận diện 10 chủ đề chính.")
st.markdown("---")

col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("📥 Nhập nội dung văn bản câu hỏi")
    
    user_input = st.text_area(
        label="Nhập chuỗi văn bản (Bằng Tiếng Anh):",
        height=180,
        value="What is the best way to learn computer programming online for free?",
        placeholder="Nhập hoặc dán đoạn văn bản cần kiểm tra vào đây..."
    )
    
    predict_btn = st.button("🚀 Tiến Hành Phân Loại", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Kết quả dự đoán từ mô hình")
    
    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Vui lòng nhập nội dung văn bản trước khi bấm phân loại!")
        else:
            with st.spinner("Đang xử lý làm sạch văn bản và dự đoán..."):
                # Bước A: Tiền xử lý làm sạch chuỗi ký tự
                cleaned = clean_text(user_input)
                
                # Bước B: Chuyển chữ sang token số nguyên và thực hiện Padding
                sequences = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
                
                # Bước C: Đẩy mảng vào mô hình dự đoán (Trả về mảng xác suất Softmax)
                predictions = model.predict(padded)[0]
                max_idx = np.argmax(predictions)
                
                # Bước D: Ánh xạ kết quả trả về nhãn text tương ứng
                try:
                    raw_label = label_encoder.inverse_transform([max_idx])[0]
                    predicted_topic = TOPIC_MAPPING.get(int(raw_label), f"Chủ đề {raw_label}")
                except:
                    predicted_topic = TOPIC_MAPPING.get(max_idx + 1, f"Chủ đề {max_idx}")
                
                confidence = predictions[max_idx] * 100

                # HIỂN THỊ KẾT QUẢ CHÍNH
                st.success(f"**Chủ đề được nhận diện:** {predicted_topic}")
                st.metric(label="Độ tin cậy chính xác (Confidence Score)", value=f"{confidence:.2f}%")
                
                # Hộp thông tin mở rộng dòng chảy dữ liệu (Pipeline) để thầy cô chấm điểm cao
                with st.expander("🔍 Chi tiết quá trình xử lý văn bản (Pipeline)"):
                    st.write(f"**1. Văn bản gốc:** `{user_input}`")
                    st.write(f"**2. Sau khi Clear & Lọc Stopwords:** `{cleaned}`")
                    st.write(f"**3. Vectơ số sau Padding (Độ dài {MAX_LENGTH}):**")
                    st.code(str(padded[0]))

                # VẼ BIỂU ĐỒ TIẾN TRÌNH XÁC SUẤT BẰNG THANH PROGRESS REAL-TIME
                st.markdown("---")
                st.write("**📊 Tỷ lệ phân bổ xác suất trên cả 10 danh mục:**")
                
                for idx, prob in enumerate(predictions):
                    topic_name = TOPIC_MAPPING.get(idx + 1, f"Chủ đề {idx+1}")
                    st.write(f"_{topic_name}_")
                    st.progress(float(prob))
                    st.caption(f"Xác suất: {prob*100:.2f}%")
    else:
        st.info("💡 Mời nhập câu test ở ô bên trái, sau đó bấm nút **Tiến Hành Phân Loại** để kiểm tra phản hồi từ mô hình LSTM!")
