# Tái hiện LinCIR và phát triển cải tiến

Project này dùng mã công bố của [LinCIR (CVPR 2024)](https://github.com/navervision/lincir) để chạy checkpoint ViT-L/14, đánh giá trên CIRR và làm baseline trước khi thử nghiệm cải tiến.

Checkpoint chính thức đã vượt qua smoke test với ảnh thật. Trên CPU/Transformers 4.57.6, đầu ra bộ mã hóa của project khớp mã gốc đã vá (max absolute difference = 0.0), xem [bằng chứng](docs/experiments/2026-09-27-upstream-parity.json). Project **chưa tái lập kết quả trong bài báo**: chưa có ảnh CIRR, điểm benchmark hoặc lượt huấn luyện đầy đủ trên GPU.

## Cách chạy nhanh nhất: Google Colab Pro

1. Tải [notebook LinCIR Colab](notebooks/LinCIR_Colab.ipynb) từ project này. Vào Google Colab → **File → Upload notebook** và chọn file vừa tải; không cần đẩy thay đổi lên GitHub trước.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy lần lượt các cell từ mục 0 đến mục 3. Các cell này gắn Google Drive, lấy đúng phiên bản mã LinCIR, tải checkpoint chính thức và chạy smoke test.
4. Chuẩn bị ảnh CIRR theo phần bên dưới, sau đó chạy mục 4. Chỉ tiếp tục khi cell báo:

   ```text
   READY 4181/4181 truy vấn, 2297/2297 ảnh trong nhãn; thiếu 0, hỏng 0 ảnh
   ```

5. Chạy mục 5 để đánh giá checkpoint công bố trên CIRR validation.
6. Sau khi baseline đánh giá thành công, có thể đổi `RUN_FULL_TRAINING = True` ở mục 6 để huấn luyện lại.
7. Đánh giá checkpoint tốt nhất ở mục 7. Muốn đối chiếu bảng test của bài báo, thêm ảnh `test1` và bật `EXPORT_CIRR_TEST` ở mục 8; notebook xuất hai JSON để bạn nộp cổng đánh giá CIRR.

Notebook tự tải nhãn CIRR val/test1 và kiểm tra SHA256. Checkpoint, cache model và log được giữ trong `MyDrive/LinCIR` khi Drive gắn thành công. Cache caption của mã tác giả nằm trong `/content/lincir/datasets` và cần tải lại nếu runtime bị xóa. Nếu gắn Drive thất bại, notebook báo dùng bộ nhớ tạm; đừng bắt đầu full training trước khi gắn Drive thành công.

Notebook khóa Transformers 4.57.6, Accelerate 1.12.0 và commit LinCIR/OpenAI CLIP. Nếu đã chạy bản notebook cũ, restart session trước khi chạy lại. Mỗi thí nghiệm dùng một `RUN_DIR`/`TRAIN_DIR` mới; log và model cũ không bị tự ghi đè.

Huấn luyện dùng toàn bộ caption CC3M + SDP, AdamW LR 1e-4, weight decay 0.01, seed 12345, 20.000 optimizer steps, batch hiệu dụng 512. Micro-batch được chọn theo VRAM, tích lũy gradient bù phần còn lại. Nếu hết VRAM, giảm `TRAIN_BATCH_SIZE` xuống một ước của 512. Colab Pro không bảo đảm chạy trọn 20.000 bước trong một phiên; checkpoint gốc chỉ lưu trọng số, **không đủ để resume chính xác optimizer/dataloader**. Notebook không tự warm-start.

Mã Phi công bố không có LayerNorm dù phần mô tả trong bài có LayerNorm. Project giữ kiến trúc mã công bố để nạp đúng checkpoint. Ngân sách 20.000 bước theo README tác giả; không tự đặt patience cho early stopping vì bài không công bố giá trị đó.

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
MyDrive/LinCIR/runs/vit_l_official/training_seed12345/checkpoints/phi_best.pt
MyDrive/LinCIR/runs/vit_l_official/training_seed12345/train.log
MyDrive/LinCIR/runs/vit_l_official/submissions/vit_l_official/
```

`pretrained_cirr_val.log` là kết quả baseline trên validation/dev. Không so trực tiếp log này với kết quả test trong bảng của bài báo.

## Chạy kiểm tra cục bộ

Yêu cầu Python 3.11 trở lên (khuyến nghị 3.11/3.12 cho Colab). Nếu clone project từ Git:

```powershell
git clone --recurse-submodules https://github.com/namPh06/XLA-.git XLA
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
python test_upstream.py
```

Kiểm tra checkpoint công bố với ảnh mẫu thật; lần đầu sẽ tải khoảng 1,8 GB model:

```powershell
python test_pretrained.py
```

Khi model đã tải xong, kiểm tra parity với hàm mã hóa gốc đã vá, không cần mạng:

```powershell
python test_pretrained.py --local-files-only --check-upstream --output output/baseline/upstream_parity.json
```

`test_upstream.py` chạy hàm trainer gốc với CLIP nhỏ và Accelerate thật; dữ liệu/điểm validation được giả lập để kiểm tra accumulation và chọn checkpoint. `test_colab_notebook.py` kiểm tra xuất submission bằng dự đoán giả lập. Các kiểm tra này không thay thế chạy notebook trên GPU với dữ liệu thật.

Smoke test chỉ chứng minh pipeline và checkpoint chạy được. Để có metric CIRR và kết luận tái lập, phải chạy notebook với đầy đủ dữ liệu trên GPU.

## Script thử nghiệm cải tiến cục bộ

Ưu tiên baseline chính thức trong notebook trước. Các script sau là **pilot riêng**: lấy tối đa 50.000 caption mỗi nguồn, khử trùng, train 1.000 bước; không tương đương lượt full training trong bài.

```powershell
python prepare_captions.py --output data/captions.jsonl --per-source 50000
python train.py --captions data/captions.jsonl --output output/pilot_baseline --steps 1000 --batch-size 16 --accumulation 32
python train.py --captions data/captions.jsonl --output output/pilot_contrastive --steps 1000 --batch-size 16 --accumulation 32 --contrastive-weight 0.1
python evaluate.py --dataset data/CIRR --checkpoint output/pilot_baseline/phi_001000.pt --output results/pilot_baseline.json
python evaluate.py --dataset data/CIRR --checkpoint output/pilot_contrastive/phi_001000.pt --output results/pilot_contrastive.json
```

Hai lệnh train mặc định dùng cùng seed 12345; script tự chọn CUDA nếu có. Máy hiện tại dùng PyTorch CPU nên chỉ thích hợp kiểm tra chức năng. InfoNCE nhìn thấy micro-batch 16, không phải 512 negative. `train.py` lưu theo bước và chưa tự chọn best checkpoint theo dev; chưa dùng kết quả pilot để tuyên bố cải tiến.

## Thành phần chính

- `third_party/lincir`: mã LinCIR chính thức, khóa tại commit đã kiểm tra.
- `notebooks/LinCIR_Colab.ipynb`: luồng chạy baseline, đánh giá và huấn luyện trên Colab.
- `lincir_core.py`, `train.py`, `evaluate.py`: phần thực nghiệm để phát triển biến thể.
- `docs/design.md`: phạm vi thực nghiệm, metric và giả thuyết cải tiến.
- `docs/experiments/`: bằng chứng smoke test đã chạy.

Mã LinCIR gốc dùng giấy phép CC BY-NC 4.0. Hãy kiểm tra giấy phép của mã, checkpoint và dữ liệu trước khi sử dụng ngoài mục đích nghiên cứu.
