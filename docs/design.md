# Đề tài: tái hiện LinCIR và kiểm chứng cải tiến

Ngày bắt đầu: 19/09/2026. Bài gốc: `5. 2312.01998v2.pdf`, CVPR 2024.
Mục tiêu người dùng: cài đặt, thực nghiệm phương pháp trong bài và cải tiến.
Môi trường đã chọn: Colab/Kaggle; máy Windows dùng kiểm tra chức năng.

## Phạm vi đợt đầu

1. Giữ mã tác giả tại `third_party/lincir`, commit
   `1dec42d118da816be2a43fd43cc0746e05f63881`, kèm LICENSE/NOTICE.
2. ViT-L/14, CLIP đóng băng, chỉ học Phi bằng caption; SMP và nhiễu
   `Uniform(0,1) * Normal(0,1)`, không chuẩn hóa embedding trước Phi.
3. Dùng nguyên lớp Phi của mã tác giả. Đây là baseline theo **mã công bố**:
   bài mô tả LayerNorm ở hai đầu, nhưng lớp Phi công bố không có LayerNorm.
4. Tạo tập pilot từ CC3M + Stable Diffusion Prompts, lưu nguồn và SHA256.
   Không dùng Midjourney dù loader hiện tại của tác giả có thêm nguồn này.
5. Biến thể đề xuất: `L = MSE + lambda * symmetric_InfoNCE`, mặc định
   lambda=0.1, temperature=0.07. Chuẩn hóa chỉ trong InfoNCE.
   Đây là giả thuyết cải tiến, chưa phải kết quả hay tuyên bố tính mới toàn ngành.
6. Đánh giá CIRR validation bằng gallery đầy đủ, loại ảnh tham chiếu,
   R@1/5/10/50 và subset R@1/2/3. Không bỏ qua ảnh thiếu/hỏng.
7. Notebook Colab/Kaggle thực hiện chuẩn bị dữ liệu, huấn luyện hai biến thể,
   đánh giá và xuất kết quả. Không tạo web app hoặc hệ thống dịch vụ.

## Thực nghiệm và giới hạn

- Thử chạy nhỏ 2 bước trước; pilot 100,000 caption, 1,000 bước.
- Một GPU: micro-batch 16, tích lũy 32 lần = batch hiệu dụng 512.
  InfoNCE chỉ nhìn thấy 16 mẫu trong micro-batch, không phải 512 negative.
- So sánh cùng dữ liệu, backbone, số bước và seed; mở rộng 3 seed khi pilot ổn.
- Chọn checkpoint theo CIRR dev R@1 như bài. Nếu dùng dev chọn lambda,
  kết quả trên dev chỉ dùng phát triển; kết luận cuối cần CIRR test hoặc
  một benchmark độc lập như FashionIQ/CIRCO, không tune trên test.
- Caption gần nghĩa vẫn có thể là false negative của InfoNCE. Khử trùng caption
  chính xác chưa xử lý được trường hợp này; theo dõi bằng thí nghiệm loại bỏ loss.
- Pilot và kiểm thử tensor không được coi là tái hiện toàn bộ bảng bài báo.
- Đợt này chưa bao gồm ViT-H/G, benchmark thứ hai, kết quả test server,
  hay huấn luyện toàn bộ dữ liệu đến hội tụ.

## Tiêu chí kiểm tra

Gradient qua encoder đóng băng phải về Phi; encoder không nhận gradient.
lambda=0 phải đúng MSE. Mẫu ghép đúng có contrastive loss thấp hơn mẫu đảo.
Metric phải đúng trên ví dụ tính tay, loại reference và không bỏ target thiếu.
Checkpoint load lại được. Notebook không chứa kết quả bịa hoặc tài khoản riêng.

## Nguồn

- https://arxiv.org/abs/2312.01998v2
- https://github.com/navervision/lincir
- https://github.com/Cuberick-Orion/CIRR
- https://huggingface.co/navervision/zeroshot-cir-models
