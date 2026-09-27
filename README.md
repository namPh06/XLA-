# Tái hiện LinCIR và phát triển cải tiến

Project này dùng mã công bố của [LinCIR (CVPR 2024)](https://github.com/navervision/lincir) để chạy checkpoint ViT-L/14, đánh giá trên CIRR và làm baseline trước khi thử nghiệm cải tiến.

Hiện tại checkpoint chính thức đã vượt qua smoke test với ảnh thật. Project **chưa tái lập kết quả trong bài báo** vì cần đủ ảnh CIRR để chạy benchmark.

## Cách chạy nhanh nhất: Google Colab Pro

1. Mở [notebook LinCIR Colab](notebooks/LinCIR_Colab.ipynb) trong Google Colab.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy lần lượt các cell từ mục 0 đến mục 3. Các cell này gắn Google Drive, lấy đúng phiên bản mã LinCIR, tải checkpoint chính thức và chạy smoke test.
4. Chuẩn bị ảnh CIRR theo phần bên dưới, sau đó chạy mục 4. Chỉ tiếp tục khi cell báo:

   ```text
   READY 4181/4181 truy vấn, 2297/2297 ảnh trong nhãn; thiếu 0, hỏng 0 ảnh
   ```

5. Chạy mục 5 để đánh giá checkpoint công bố trên CIRR validation.
6. Sau khi baseline đánh giá thành công, có thể đổi `RUN_FULL_TRAINING = True` ở mục 6 để huấn luyện lại.

Notebook tự tải nhãn CIRR và kiểm tra SHA256. Checkpoint, cache và log được giữ trong `MyDrive/LinCIR` để không mất khi Colab ngắt phiên.

## Lấy ảnh CIRR

Ảnh CIRR có nguồn từ NLVR2 và không được phát hành công khai trực tiếp. Cần đăng ký bằng thông tin của chính bạn:

1. Điền [form xin quyền truy cập ảnh NLVR2](https://docs.google.com/forms/d/e/1FAIpQLSdB_OhgmpQULV17kjQ4iitftILbOJjuGgJ2ECmg-HdmkjUSAg/viewform) và đồng ý điều khoản dùng cho nghiên cứu. Xem [hướng dẫn NLVR2](https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download).
2. Sau khi được cấp quyền, làm theo [hướng dẫn tải ảnh CIRR](https://cirr.zheyuanliu.me/raw-image-download), tải archive và kiểm tra checksum mà CIRR cung cấp.
3. Giải nén, giữ nguyên tên file và đưa thư mục `dev` lên Google Drive tại:

   ```text
   MyDrive/datasets/CIRR/dev/
   ```

4. Chạy mục 4 trong notebook. Cell này tự tải hai file nhãn validation vào thư mục `cirr` và kiểm tra toàn bộ ảnh thiếu hoặc hỏng.

Cấu trúc cuối cùng:

```text
MyDrive/datasets/CIRR/
├── cirr/
│   ├── captions/cap.rc2.val.json
│   └── image_splits/split.rc2.val.json
└── dev/
    ├── dev-0-0-img0.png
    └── ...
```

Không đổi tên ảnh và không báo cáo kết quả nếu kiểm tra dữ liệu vẫn là `NOT_READY`.

## Kết quả được lưu ở đâu

```text
MyDrive/LinCIR/checkpoints/lincir_large.pt
MyDrive/LinCIR/runs/vit_l_official/cirr_readiness.json
MyDrive/LinCIR/runs/vit_l_official/pretrained_cirr_val.log
MyDrive/LinCIR/runs/vit_l_official/checkpoints/phi_best.pt
```

`pretrained_cirr_val.log` là kết quả baseline trên validation/dev. Không so trực tiếp log này với kết quả test trong bảng của bài báo.

## Chạy kiểm tra cục bộ

Yêu cầu Python 3.10 trở lên. Nếu clone project từ Git:

```powershell
git clone --recurse-submodules <URL_PROJECT>
cd XLA
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Chạy các kiểm tra chức năng:

```powershell
python test_core.py
python test_smoke.py
python test_colab_notebook.py
```

Kiểm tra checkpoint công bố với ảnh mẫu thật; lần đầu sẽ tải khoảng 1,8 GB model:

```powershell
python test_pretrained.py
```

Smoke test chỉ chứng minh pipeline và checkpoint chạy được. Để có metric CIRR và kết luận tái lập, phải chạy notebook với đầy đủ dữ liệu trên GPU.

## Thành phần chính

- `third_party/lincir`: mã LinCIR chính thức, khóa tại commit đã kiểm tra.
- `notebooks/LinCIR_Colab.ipynb`: luồng chạy baseline, đánh giá và huấn luyện trên Colab.
- `lincir_core.py`, `train.py`, `evaluate.py`: phần thực nghiệm để phát triển biến thể.
- `docs/design.md`: phạm vi thực nghiệm, metric và giả thuyết cải tiến.
- `docs/experiments/`: bằng chứng smoke test đã chạy.

Mã LinCIR gốc dùng giấy phép CC BY-NC 4.0. Hãy kiểm tra giấy phép của mã, checkpoint và dữ liệu trước khi sử dụng ngoài mục đích nghiên cứu.
