# Hệ thống Nhận diện Hành vi Học tập (Classroom Behavior Detector)

## Giới thiệu

"Hệ thống Nhận diện Hành vi Học tập" là một giải pháp dựa trên thị giác máy tính được thiết kế để tự động phát hiện và phân tích các hành vi của học sinh trong môi trường lớp học. Hệ thống ứng dụng các mô hình phát hiện đối tượng YOLO (You Only Look Once) tiên tiến, cụ thể là các phiên bản YOLOv8, YOLOv9 và YOLOv10, để nhận dạng các hành vi cụ thể và thực hiện so sánh hiệu suất giữa các biến thể mô hình này.

Mục tiêu cốt lõi là phát triển một công cụ trực quan, dễ sử dụng cho các nhà giáo dục, cho phép họ phân tích hành vi trong lớp thông qua hình ảnh và video, từ đó cung cấp những hiểu biết giá trị để hỗ trợ quản lý lớp học và nâng cao chất lượng dạy và học.

**Từ khóa:** nhận diện hành vi, hành vi học sinh, thị giác máy tính, phát hiện đối tượng, YOLO, phân tích lớp học, công nghệ giáo dục, học sâu, quản lý lớp học.

## Các tính năng chính

* **Phát hiện và Phân loại Hành vi:** Tự động phát hiện và phân loại ba hành vi chính của học sinh: giơ tay phát biểu (hand-raising), đọc bài (reading), và viết bài (writing).
* **So sánh hiệu suất mô hình YOLO:** Nghiên cứu và so sánh hiệu quả của các phiên bản YOLO khác nhau (v8, v9, v10) để tìm ra mô hình tối ưu cho bài toán.
* **Giao diện Phân tích Trực quan:** Cung cấp giao diện web thân thiện với người dùng (sử dụng Gradio) để dễ dàng tải lên và phân tích hình ảnh/video.
    * **Suy luận trên ảnh (Image Inference):** Phát hiện hành vi trong ảnh tĩnh.
    * **Suy luận trên video (Video Inference):** Xử lý video để xác định hành vi trong từng khung hình.

## Mô hình và Bộ dữ liệu

* **Mô hình triển khai chính:** `yolov9e_fold3` (phiên bản YOLOv9e được huấn luyện trên fold3 của bộ dữ liệu).
* **Các mô hình được nghiên cứu:** YOLOv8x, YOLOv9e, YOLOv10x.
* **Bộ dữ liệu:** SCB-Dataset3-S, tập trung vào 3 hành vi: giơ tay, đọc sách, viết bài. Dữ liệu được phân chia bằng K-Fold Cross-Validation (K=5) để chọn ra fold tối ưu cho việc huấn luyện mô hình cuối cùng.

## Kết quả nổi bật (trên SCB-Dataset3-S với yolov9e_fold3)

* **Precision (P):** 72.2%
* **Recall (R):** 73.0%
* **mAP@50:** 77.8%
* **mAP@50:95:** 61.1%
* **Params (M):** 56.3
* **GFLOPs:** 189.0

Mô hình `yolov9e_fold3` cho thấy hiệu suất vượt trội so với các phiên bản YOLOv5 và YOLOv8 được huấn luyện trong nghiên cứu tham chiếu trên cùng bộ dữ liệu, và có hiệu suất cạnh tranh khi so sánh với các biến thể YOLOv7.

## Công nghệ và Thư viện sử dụng

* **ultralytics:** Khung làm việc (framework) YOLO cho mô hình và chức năng dự đoán.
* **torch/torchvision:** Khung làm việc học sâu PyTorch.
* **gradio:** Thư viện tạo giao diện web tương tác.
* **opencv-python:** Thư viện xử lý ảnh và video.
* **matplotlib:** Công cụ trực quan hóa dữ liệu.
* **wandb (Weights & Biases):** Công cụ theo dõi thí nghiệm (được đề cập trong `requirements.txt`).
* **Python 3.10+**
* **Git**

## Thiết lập Môi trường Phát triển

**Điều kiện tiên quyết:**
* Python 3.10+
* Git
* GPU có hỗ trợ CUDA (khuyến nghị cho việc huấn luyện và suy luận nhanh hơn)

**Các bước cài đặt:**
1.  Sao chép (Clone) kho lưu trữ:
    ```bash
    git clone [https://github.com/KhongDucQuang/btl_ds.git](https://github.com/KhongDucQuang/btl_ds.git)
    cd Classroom-Behavior-Detector
    ```
2.  Cài đặt các thư viện phụ thuộc:
    ```bash
    pip install -r requirements.txt
    ```

## Chạy Ứng dụng (Giao diện Gradio)

1.  Đảm bảo tất cả các thư viện phụ thuộc đã được cài đặt (xem bước trên).
2.  Đảm bảo tệp mô hình đã huấn luyện  nằm trong thư mục chính của dự án (hoặc theo đường dẫn được cấu hình trong `app.py`).
3.  Chạy ứng dụng:
    ```bash
    python app.py
    ```
4.  Truy cập giao diện thông qua URL cục bộ được cung cấp trong terminal (thường là `http://127.0.0.1:7860`) trên trình duyệt web.

## Kiến trúc Giao diện Gradio

Ứng dụng Gradio được cấu trúc bằng giao diện dạng thẻ, trong đó mỗi thẻ đại diện cho một chế độ suy luận khác nhau (Ảnh hoặc Video). Mỗi thẻ là một phiên bản Giao diện Gradio riêng biệt với các thành phần đầu vào và đầu ra cụ thể.

### Quy trình suy luận trên ảnh:
1.  Tải ảnh đầu vào bằng OpenCV.
2.  Chạy dự đoán YOLO trên ảnh.
3.  Trích xuất các hộp giới hạn, nhãn lớp, và điểm tin cậy.
4.  Vẽ các hình chữ nhật có màu xung quanh đối tượng được phát hiện.
5.  Thêm nhãn văn bản với tên lớp và điểm tin cậy.
6.  Chuyển đổi ảnh từ BGR sang RGB để hiển thị trên Gradio.

### Quy trình suy luận trên video:
1.  Mở video đầu vào và lấy các thuộc tính (chiều rộng, chiều cao, fps).
2.  Tạo đối tượng ghi video cho đầu ra với các thuộc tính tương tự.
3.  Với mỗi khung hình:
    * Chạy dự đoán YOLO.
    * Vẽ hộp giới hạn và nhãn.
    * Ghi khung hình đã xử lý vào video đầu ra.
4.  Giải phóng tài nguyên.
5.  Trả về đường dẫn đến video đã xử lý.

## Ánh xạ màu cho các lớp

| Class ID | Color (BGR)   | Description (Hành vi) |
| :------- | :------------ | :-------------------- |
| 0        | (209, 54, 40) | Giơ tay (Hand-raising) |
| 1        | (37, 194, 45) | Đọc bài (Reading)      |
| 2        | (34, 92, 240) | Viết bài (Writing)     |



