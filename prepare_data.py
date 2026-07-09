import ember
import os
import numpy as np

# Đường dẫn CHÍNH XÁC trỏ thẳng vào thư mục chứa các file .jsonl sau khi giải nén
EMBER_DIR = r"C:\new\malware_project\ember_data\ember2018"

def main():
    print(f"[*] Đang kiểm tra dữ liệu tại: {EMBER_DIR}")
    if not os.path.exists(EMBER_DIR):
        print("[!] LỖI: Không tìm thấy thư mục. Hãy kiểm tra lại đường dẫn giải nén.")
        return
        
    # Kiểm tra xem file X_train.dat đã được tạo chưa
    if not os.path.exists(os.path.join(EMBER_DIR, "X_train.dat")):
        print("[*] BƯỚC 1: Bắt đầu trích xuất đặc trưng (Vectorization)...")
        print("[!] LƯU Ý: Quá trình này sẽ ngốn khoảng 10-12GB RAM và mất từ 15-30 phút.")
        print("[!] Máy tính có thể bị lag, vui lòng không tắt cửa sổ CMD!")
        # Hàm này sẽ parse toàn bộ 8GB JSONL để tạo ra các file .dat
        ember.create_vectorized_features(EMBER_DIR, 1) 
        print("[*] Trích xuất thành công các file .dat!")
    else:
        print("[*] File .dat đã tồn tại, tự động chuyển sang bước Load.")

    print("\n[*] BƯỚC 2: Load thử tập dữ liệu Test vào RAM...")
    X_test, y_test = ember.read_vectorized_features(EMBER_DIR, subset="test")
    
    # Lọc bỏ các mẫu không có nhãn (unlabeled)
    mask = y_test != -1
    
    print("\n=== BÁO CÁO NGHIỆM THU DỮ LIỆU ===")
    print(f"Tổng số mẫu Test có nhãn : {mask.sum():,} mẫu")
    print(f"Kích thước ma trận (Shape): {X_test[mask].shape} (2.381 chiều đặc trưng)")
    print(f"Số mẫu Malware (Nhãn 1)   : {y_test[mask].sum():,} mẫu")
    print(f"Số mẫu Clean (Nhãn 0)     : {(y_test[mask]==0).sum():,} mẫu")
    print("====================================")
    print("\n[✓] HOÀN TẤT TUYỆT ĐỐI! Dữ liệu đã sẵn sàng cho Tầng 1 (ML Triage).")

if __name__ == "__main__":
    main()