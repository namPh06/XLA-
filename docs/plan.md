# Kế hoạch triển khai LinCIR

**Goal:** có mã chạy được cho baseline, biến thể loss và đánh giá CIRR.
**Architecture:** tái dùng Phi chính thức; script nhỏ cho dữ liệu, train và eval.
**Tech Stack:** Python, PyTorch, Transformers, spaCy, Hugging Face Datasets.
**Spec:** [design.md](design.md).

## Công việc

- [x] Đọc bài, lấy mã chính thức, kiểm tra máy và chọn Colab/Kaggle.
- [ ] `test_core.py`: kiểm tra gradient, loss và metric trước khi cài đặt.
- [ ] `lincir_core.py`: chèn pseudo-token, SMP loss, contrastive loss, CIRR metrics.
- [ ] `prepare_captions.py`: lấy caption, mask theo POS, khử trùng và lưu nguồn.
- [ ] `train.py`: huấn luyện một GPU/CPU, gradient accumulation, log và checkpoint.
- [ ] `evaluate.py`: gallery CIRR đầy đủ, metric và kết quả JSON.
- [ ] `notebooks/LinCIR_Colab_Kaggle.ipynb`, README: hướng dẫn chạy từng bước.
- [ ] Chạy kiểm thử CPU, smoke train/eval trên mô hình CLIP nhỏ ngẫu nhiên.
- [ ] Review độc lập, sửa lỗi, đóng gói mã để chuyển sang Colab/Kaggle.

## Review focus

Không normalize đầu vào Phi; không vô tình dùng no_grad cho nhánh pseudo-token;
không để reference trong xếp hạng; thiếu ảnh phải báo lỗi; dữ liệu pilot và số đo
trên tensor giả phải được ghi nhãn rõ. Không dùng test labels chọn siêu tham số.

Kiểm tra chính: `python test_core.py`; smoke sẽ dùng CLIP config nhỏ offline,
không tải backbone lớn về laptop và không tuyên bố chất lượng truy hồi từ smoke.
