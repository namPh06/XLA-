# Kế hoạch triển khai LinCIR

**Goal:** có mã chạy được cho baseline, biến thể loss và đánh giá CIRR.
**Architecture:** tái dùng Phi chính thức; script nhỏ cho dữ liệu, train và eval.
**Tech Stack:** Python, PyTorch, Transformers, spaCy, Hugging Face Datasets.
**Spec:** [design.md](design.md).

## Công việc

- [x] Đọc bài, lấy mã chính thức, kiểm tra máy và chọn Colab/Kaggle.
- [x] `test_core.py`: kiểm tra gradient, loss, pseudo-token và metric.
- [x] `lincir_core.py`: chèn pseudo-token, SMP loss, contrastive loss, CIRR metrics.
- [x] `prepare_captions.py`: mã lấy caption, mask theo POS, khử trùng và lưu nguồn (chưa chạy tải toàn bộ).
- [x] `train.py`: huấn luyện một GPU/CPU, gradient accumulation, log và checkpoint; đã kiểm tra CPU.
- [x] `evaluate.py`: gallery CIRR đầy đủ, metric và kết quả JSON; đã kiểm tra dữ liệu giả lập.
- [x] `notebooks/LinCIR_Colab.ipynb`, README: hướng dẫn từng bước, bảo vệ run cũ.
- [x] Kiểm thử CPU, smoke train/eval với CLIP nhỏ; vá và kiểm tra trainer gốc với Accelerate thật.
- [x] Smoke ảnh thật với checkpoint công bố; parity với hàm mã hóa gốc đã vá.
- [ ] Cấp quyền/tải đủ ảnh CIRR và chạy notebook trên Colab GPU.
- [ ] Ghi nhận baseline CIRR dev và điểm test từ máy chủ.
- [ ] Huấn luyện toàn bộ CC3M + SDP, lưu config/log/checkpoint và đánh giá.
- [ ] So sánh cải tiến cùng điều kiện, nhiều seed; kết luận dựa trên số đo.

## Review focus

Không normalize đầu vào Phi; không vô tình dùng no_grad cho nhánh pseudo-token;
không để reference trong xếp hạng; thiếu ảnh phải báo lỗi; dữ liệu pilot và số đo
trên tensor giả phải được ghi nhãn rõ. Không dùng test labels chọn siêu tham số.

Kiểm tra chính: `python test_core.py`, `python test_smoke.py`,
`python test_colab_notebook.py`, `python test_upstream.py`.
Backbone lớn đã tải để chạy `test_pretrained.py`; smoke không đo chất lượng truy hồi.
