# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU (chạy lại lõi notebook trên máy local theo đúng code cell 2–14; không dùng GPU)

**Python / PyTorch / Ultralytics:** Python 3.11.9 / PyTorch 2.15.0+cpu / Ultralytics 8.4.145 (đúng phiên bản ghim `8.4.145`)

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (sha256 lần lượt `c62d41bf9625…`, `0ebbc80d4a76…`, `55ed65c56c91…` — khớp checksum notebook kiểm tra)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ) — **không đổi**.

**Thay đổi so với notebook nguồn:** Không đổi checkpoint, threshold hay mã nguồn của các cell task. Khác biệt duy nhất: ba tác vụ được chạy lại trên máy local (CPU) với code sao chép nguyên văn từ notebook thay vì Google Colab; ảnh COCO và checkpoint đã qua kiểm tra SHA-256 như notebook; ô validation (cell 14) chạy lại và báo `PASS`.

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=**468** ; `class_name`=**cab** ; `rank`=1 ; `score`=**0.510914** ; `taxonomy_name`=**ImageNet-1K**.
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   — Nói về **cả bức ảnh**: classification đầu ra là một nhãn cấp ảnh (image-level), không có `bbox_xyxy` hay `polygon_xy` trong record như detection/segmentation; record cũng không có `instance_id`. Đó là lý do JSON chỉ có `rank`, `score`, `class_id/class_name`. Trong ảnh `traffic` thực tế có rất nhiều xe bus, ô tô, người đi bộ — model chỉ được phép gán **một** lớp cho toàn cảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   — Đi kèm **taxonomy của checkpoint** (`yolo11n-cls.pt` dùng ImageNet-1K, 1000 lớp). Model chỉ chọn trong danh sách cố định này; nó không thể trả ra tên lớp ngoài danh sách. Vì ảnh là cảnh COCO còn taxonomy là ImageNet nên top-1 là `cab` với score chỉ 0.51 — trông “lạ” so với cảnh nhiều xe bus, đó là giới hạn của taxonomy chứ không phải lỗi máy.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   — Vì tên lớp chỉ có ý nghĩa **trong phạm vi taxonomy** của nó. `class_id` là mã số duy nhất dùng máy đọc được (tránh lỗi trùng tên/chính tả), `class_name` để người đọc, còn `taxonomy_name` cho biết id/tên này thuộc danh sách nào (ImageNet-1K hay COCO-80) — cùng một id số ở hai taxonomy khác nhau sẽ chỉ hai lớp khác nhau. Thiếu `taxonomy_name` thì không tái lập được cấu hình model đã tạo output.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   — Guideline cần quy định: (a) tiêu chí chọn lớp chính cho single-label (ví dụ “lớp chiếm diện tích/lượng pixel lớn nhất”, hoặc “chủ đề chính mà ảnh chụp để làm gì”); (b) thứ tự ưu tiên khi có nhiều lớp thỏa (priority list); (c) xử lý khi không lớp nào phù hợp (thêm lớp “other/unknown” hay escalation). Ảnh `traffic` là ví dụ: nhiều bus và ô tô cùng lúc — nếu không có quy tắc, hai annotator sẽ chọn khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
   — `score` chỉ là **model score**: độ tin cậy của model khi xếp hạng prediction, không phải độ chính xác đã được người kiểm chứng. Top-1 `cab` 0.51 vẫn có thể sai so với cảnh thực; score cao không chứng minh nhãn đúng, score thấp không chứng minh nhãn sai. Ground truth phải do người gán theo guideline, còn model score chỉ dùng để xếp hạng/lọc prediction.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng (output ô so sánh trong notebook, sample `kitchen`, ảnh 640×427):

- `threshold=0.20` → **17** prediction
- `threshold=0.35` → **11** prediction (ngưỡng của JSON)
- `threshold=0.60` → **6** prediction

1. Chép một record (lớp `person`, score cao nhất @0.35): `class_name`=**person** ; `score`=**0.912624** ; `bbox_xyxy`=[**385.33, 69.24, 498.92, 348.92**] ; `bbox_width`=**113.58** ; `bbox_height`=**279.68**.
2. Giải thích box bằng lời thường: góc trên-trái ở điểm **(385.33, 69.24)** — cách mép trái 385 pixel và cách mép trên 69 pixel (gốc `(0,0)` ở góc trên-trái của ảnh 640×427); góc dưới-phải ở **(498.92, 348.92)**. Hộp rộng **113.58 pixel** và cao **279.68 pixel**, ôm quanh người đầu bếp đứng bên phải ảnh. Cả tọa độ đều theo **pixel**, đơn vị ghi trong trường `coordinate_unit="pixel"`, định dạng `bbox_format="xyxy"`.
3. Chọn hai ngưỡng: `0.20` có **17** prediction; `0.60` còn **6** prediction. Hạ ngưỡng giữ thêm các hộp score thấp (spoon, potted plant, dining table, bottle…), nâng ngưỡng chỉ giữ các hộp score cao (person, bowl, oven).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   — Phải xem và quyết định nhiều hộp score thấp hơn: loại hộp trùng lặp (NMS không gộp hết, vd. nhiều `bowl` chồng nhau trên bàn), phát hiện false positive (vật bị gán nhầm lớp), và ghi nhận vật thật bị box lệch. Nói cách khác, hạ ngưỡng tăng recall nhưng tăng khối lượng QC/rework — threshold là cấu hình **lọc prediction**, không phải quy tắc quyết định vật nào được phép tồn tại khi tạo ground truth.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):
   — “Box phải bao trọn phần vật nhìn thấy được (tính cả phần bị che trong suốt); nếu vật bị cắt mép ảnh, cạnh box kéo tới đúng mép ảnh (x_min/y_min = 0 hoặc x_max/y_max = kích thước ảnh); mỗi vật thể riêng biệt có một box riêng, hai vật cùng lớp vẫn hai box khác nhau.”
6. Trên ảnh `kitchen`, vật bị che khuất/cắt mép: record `person` score **0.610931**, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] — chỉ thấy một phần cánh tay/bàn tay của một người bị cắt ở **mép trái ảnh**, box bị kẹp về x_min ≈ 0.
   Tôi sẽ xử lý **hai tầng**: phần “vật cắt mép → box chạm mép ảnh, vẫn gán class nếu đủ đặc trưng nhận diện” là quy tắc guideline áp dụng được cho nhiều ảnh; nhưng trường hợp **chỉ thấy một phần chi rất nhỏ, không chắc là người** thì rule không phủ được → đưa lên **escalation** hỏi Lab Coach, không tự bịa nhãn, vì quyết định “đếm hay không đếm vật này” sẽ ảnh hưởng nhất quán cả dataset.
   Lỗi khác nhìn thấy trên PNG: box `oven` score 0.6343 bên phải ảnh [489.09, 200.73, 617.81, 344.88] phủ vùng **bồn rửa/kệ dưới chảo treo** — nhiều khả năng gán nhầm lớp (cấu trúc này trông giống sink/counter hơn là oven). Ghi nhận như quan sát lỗi prediction, không sửa nhãn cho “đúng model”.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record (`kitchen-001`): `instance_id`=**kitchen-001** ; `class_name`=**person** ; `score`=**0.899318** ; `polygon_point_count`=**348** ; 3–5 điểm đầu của `polygon_xy` = **[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]**.
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   — Box chỉ là hình chữ nhật bao ngoài (người `kitchen-001` chiếm cả khoảng trống hai bên vai/hông không thuộc người); polygon 348 điểm bám theo **biên thật của instance**: cong theo vai, tạp dề, chân — cho biết chính xác pixel nào thuộc vật. Vì vậy polygon phân biệt được hai vật chồng nhau/kề nhau tốt hơn box, nhưng tốn nhiều điểm dữ liệu hơn (348 cặp `[x, y]` so với 4 số của box).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   — `instance_id` (`kitchen-001`, `kitchen-002`…) chỉ nhận diện **từng instance riêng lẻ trong bộ output của bài lab** để phân biệt hai vật cùng lớp (vd. `kitchen-002` và `kitchen-003` đều là `bowl` nhưng hai instance khác nhau). Nó không phải class ID vì class của instance nằm ở `class_id`/`class_name` (nhiều instance có thể chung một class ID); cũng không phải tracking ID vì không có khái niệm “cùng một vật qua nhiều frame/video” ở đây — mỗi ảnh được xử lý độc lập.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo:
   — “Vẽ polygon bám sát biên nhìn thấy được của instance tại thời điểm chụp; điểm đặt trên mép ngoài của vật (không lấn sang vật nền); khi hai vật chạm/che nhau, biên của vật phía trước vẽ theo đường che khuất nhìn thấy được; mỗi instance một polygon riêng, không gộp hai vật cùng lớp thành một vùng.”
5. Trên ảnh `kitchen`, chỗ biên mờ / hai vật chạm nhau / bị che: mask `dining table` (`kitchen-005`, 598 điểm) phủ lên bàn và **chồng vùng với box/mask `oven`** (`kitchen-007`) ở phía trái — ranh giới mặt bàn và lò nướng bị viền che khuất (khăn, vật dụng treo) nên biên mask lẫn sang nhau. Ngoài ra hai instance `spoon` (`kitchen-008`, `kitchen-011`) là những chiếc **vá múc (ladle)** treo tường — tên lớp `spoon` chỉ “gần đúng”, ranh giới spoon/ladle trong taxonomy COCO-80 là điểm mơ hồ về lớp, không phải về biên.
   Tôi chọn: phần “biên vẽ theo đường che khuất nhìn thấy được” để vào **guideline** vì là quy tắc lặp lại được; còn “spoon hay ladle, và biên dining table/oven đâu là đúng” là ca quyết định nhãn khó, rule chưa phủ → **escalation** hỏi Lab Coach trước khi gán, tránh tự bịa nhãn cho cả dataset.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh, chọn trong taxonomy cố định (vd. ImageNet-1K), có guideline single-label | Ảnh `traffic` có nhiều lớp cùng lúc (bus, car, person) nhưng taxonomy ImageNet cho top-1 `cab` 0.51 — nhãn cấp ảnh không khớp cảnh đa vật | Đọc guideline chọn-lớp-chính, gán đúng một lớp theo priority, ca không rõ thì escalation | So nhãn với guideline (đúng priority?), rà các nhãn hai annotator không đồng thuận |
| Phát hiện vật thể | Một cặp (class, box `xyxy` pixel) cho **mỗi vật thể**, kể cả vật bị che/cắt mép theo quy tắc | `oven` 0.63 box phủ nhầm vùng sink; nhiều box `bowl` chồng nhau 0.38–0.72; `person` 0.61 chỉ là cánh tay cắt mép | Vẽ box đủ vật theo quy tắc box (biên cắt mép → chạm mép ảnh), không dùng threshold để bỏ vật | Kiểm tra thiếu vật (missed), box lệch, lớp sai; đối chiếu với prediction nhưng không chép prediction làm ground truth |
| Instance segmentation | Một instance = (class, polygon biên pixel), hai vật cùng lớp vẫn hai instance riêng | Mask `dining table` chồng vùng `oven`; hai `spoon` thực là ladle; biên bị che khuất (khăn treo) làm polygon lấn sang vật bên cạnh | Vẽ polygon từng instance bám biên nhìn thấy được, tách instance khi hai vật chạm nhau | Rà polygon thủng/lấn biên, instance bị gộp/tách sai, tên lớp mơ hồ → escalation |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   — Chỉ dùng đúng ba ảnh COCO công khai có sẵn trong notebook (đã kiểm tra checksum, có `IMAGE_ATTRIBUTION.md`); không tải lên Colab/GitHub bất kỳ ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ nào, và không ghi thông tin cá nhân vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   — Dừng ngay cell đang chạy, không đưa dữ liệu đó vào pipeline, và báo **Lab Coach** (kênh hỗ trợ của lớp); nếu đã lỡ upload lên nơi công khai thì báo thêm GV/Lab Coach để xử lý trước khi nộp link repo.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS` (đã chạy lại logic validation cell 14 trên local: `PASS: đủ 3 JSON, 3 PNG, attribution và instance_id duy nhất.`)
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS *(còn lại cho người nộp bài)*
