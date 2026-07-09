import streamlit as st
import xgboost as xgb
import numpy as np
np.int = int
np.bool = bool
np.float = float
import requests
import json
import time
import lief

# Vá lỗi tương thích phiên bản cho thư viện ember cũ
# Vá thêm lỗi read_out_of_bound vào danh sách
for attr in ['bad_format', 'bad_file', 'pe_error', 'parser_error', 'read_out_of_bound']:
    if not hasattr(lief, attr):
        setattr(lief, attr, Exception)
import ember

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hybrid SOC Pipeline", page_icon="🛡️", layout="wide")
st.title("🛡️ SOC Hybrid Pipeline: Phân tích Mã độc Tĩnh")
st.markdown("Hệ thống triage tự động sử dụng **XGBoost** kết hợp phân tích chuyên sâu bằng **Llama 3.2**.")

# --- HÀM TẢI MÔ HÌNH (CACHE ĐỂ TRÁNH TRÀN RAM) ---
@st.cache_resource
def load_xgboost_model():
    try:
        model = xgb.Booster()
        # Đảm bảo file model này nằm cùng thư mục hoặc trỏ đúng đường dẫn
        model.load_model("xgboost_ember_layer1.json") 
        return model
    except Exception as e:
        st.error(f"Lỗi tải model XGBoost: {e}")
        return None

xgb_model = load_xgboost_model()

# --- HÀM GỌI OLLAMA API (LLAMA 3.2 LOCAL) ---
def analyze_with_llama(risk_score, top_features_text):
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""
    Bạn là một chuyên gia phân tích SOC (Tier-2 Analyst). 
    Một file PE vừa được XGBoost chấm điểm rủi ro là {risk_score:.3f} (Vùng xám).
    
    Dưới đây là Top các đặc trưng đáng ngờ nhất theo phân tích SHAP:
    {top_features_text}
    
    Dựa vào các đặc trưng tĩnh này, hãy:
    1. Kết luận file này là MALWARE hay BENIGN.
    2. Giải thích ngắn gọn lý do bằng tiếng Việt (tối đa 3 câu).
    3. Gợi ý 1-2 kỹ thuật MITRE ATT&CK có thể liên quan.
    """
    
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "Không nhận được phản hồi từ mô hình.")
    except Exception as e:
        return f"Lỗi kết nối đến Ollama: {e}. Vui lòng kiểm tra xem Ollama đã chạy chưa."

# --- GIAO DIỆN CHÍNH ---
uploaded_file = st.file_uploader("Tải lên file vector đặc trưng (.dat) hoặc chọn file mẫu để phân tích", type=['dat', 'exe'])

if uploaded_file is not None:
    st.info("Đang trích xuất vector đặc trưng EMBER (2.351 chiều) trực tiếp từ file thô...")
    
    try:
        # --- BẮT ĐẦU ĐOẠN CODE BẠN VỪA HỎI ---
        bytez = uploaded_file.read()
        extractor = ember.PEFeatureExtractor(2)
        raw_features = np.array(extractor.feature_vector(bytez), dtype=np.float32)
        real_features = raw_features.reshape(1, -1)[:, :2351] 
        # --- KẾT THÚC ---

        with st.spinner("Đang phân loại qua XGBoost (Tầng 1)..."):
            # Ném thẳng đặc trưng thật vào XGBoost
            dmatrix = xgb.DMatrix(real_features)
            risk_score = float(xgb_model.predict(dmatrix)[0]) if xgb_model else 0.55
            #risk_score = 0.55
    except Exception as e:
        st.error(f"Lỗi: Không thể phân tích file này. Hãy chắc chắn đây là file PE hợp lệ (.exe, .dll). Chi tiết lỗi: {e}")
        st.stop() # Dừng app tại đây nếu file bị lỗi

    st.subheader("Tầng 1: Phân loại nhanh (ML Triage)")
    
    # ... (Giữ nguyên phần code hiển thị col1, col2 và if-else gọi Tầng 2 LLM như cũ) ...
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="XGBoost Risk Score", value=f"{risk_score:.4f}")
    
    with col2:
        if risk_score < 0.3:
            st.success("🟢 KẾT LUẬN: AN TOÀN (BENIGN) - File được đưa vào Whitelist.")
        elif risk_score > 0.7:
            st.error("🔴 KẾT LUẬN: MÃ ĐỘC (MALWARE) - Kích hoạt quy trình cách ly ngay lập tức.")
        else:
            st.warning("🟡 KẾT LUẬN: VÙNG XÁM (GREY ZONE) - Điểm số không chắc chắn. Chuyển tiếp sang Tầng 2.")

    # --- KÍCH HOẠT TẦNG 2 NẾU RƠI VÀO VÙNG XÁM ---
    if 0.3 <= risk_score <= 0.7:
        st.divider()
        st.subheader("Tầng 2: Phân tích chuyên sâu (Llama 3.2)")
        
        # Mock dữ liệu SHAP (thay bằng hàm trích xuất SHAP thật của bạn)
        mock_shap_features = """
        - header_11 (size_of_code): Lớn bất thường, dấu hiệu bị packed.
        - section_242: Thuộc tính Write + Execute (W+X) được bật đồng thời.
        - import_122: Gọi hàm VirtualAlloc và CreateRemoteThread.
        """
        
        with st.spinner("Đang truy xuất SHAP values và phân tích ngữ nghĩa qua Llama 3.2... (Quá trình này có thể mất ~15 giây)"):
            llm_result = analyze_with_llama(risk_score, mock_shap_features)
            
        st.markdown("### Báo cáo từ Trợ lý AI (SOC Tier-2):")
        st.info(llm_result)