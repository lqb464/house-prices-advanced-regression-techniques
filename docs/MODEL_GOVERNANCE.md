# Quản trị model

StocKast xem output của notebook là nguồn số liệu chuẩn cho từng ticker/run. Việc chọn component hoặc hybrid ensemble được khóa trên tập validation; holdout chỉ được dùng một lần làm quality gate cho run đó. Gate không đạt chỉ ảnh hưởng promotion của artifact/run tương ứng và không phải kết luận cho framework hay ticker khác.

## Các promotion gate

- MAE trên holdout không được tệ hơn zero-return baseline của ticker đang đánh giá.
- Interval coverage mục tiêu 80%, được calibration trên validation, phải đạt ít nhất 70% trên holdout.
- Artifact checksum, schema test, raw-to-prediction test và API test đều phải đạt.
- Feature drift gần nhất, realized MAE và interval coverage phải được rà soát trước khi promote version mới.

## Rà soát thủ công

Inference được gắn cờ khi interval của kịch bản cắt qua 0, input nằm ngoài thống kê tham chiếu, schema coercion sinh cảnh báo hoặc chính artifact hybrid đang ở trạng thái `deployment_blocked`.

## Ngoài phạm vi

Service không đặt lệnh, đưa khuyến nghị đầu tư, suy luận mức độ phù hợp hay khẳng định quan hệ nhân quả từ các technical indicator. Kết quả của một ticker ví dụ không được suy rộng tự động sang ticker khác.
