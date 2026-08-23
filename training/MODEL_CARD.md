# StocKast — hybrid ensemble candidate nghiên cứu

## Mục đích sử dụng

Ước lượng cumulative log-return theo horizon cấu hình của một ticker sau khi thị trường đóng cửa. `VIC.VN` chỉ là ticker ví dụ của artifact hiện tại; output hỗ trợ nghiên cứu model, error analysis và monitoring. Không được dùng output để đặt lệnh, xác định quy mô danh mục hoặc trình bày như lời khuyên đầu tư.

## Dữ liệu và cách chia tập

- Run minh hoạ dùng adjusted OHLCV ngày từ Yahoo Finance, từ 2018-01-01 đến 2026-08-14.
- 2.246 phiên thô; 2.192 dòng feature có label sau rolling window và bước tạo target.
- Chia theo thời gian: 60% train, 20% validation và 20% frozen holdout.
- Holdout: 2024-12-03 đến 2026-08-07, gồm 439 dự báo.

## Component và hybrid ensemble

Candidate kết hợp GRU với Ridge, Extra Trees và Histogram Gradient Boosting. Các component được benchmark độc lập, sau đó kết hợp bằng hybrid ensemble trên validation (MAE **0,03071** so với **0,02996** của GRU trong run ví dụ); kết quả này chỉ đại diện cho run ticker ví dụ và cần được tái lập cho ticker khác.

## Kết quả frozen holdout

- GRU MAE: 0,07313.
- Zero-return baseline MAE: 0,06067.
- Directional accuracy: 40,32%.
- Interval coverage mục tiêu 80%, calibration trên validation: 43,74%.

## Trạng thái

Artifact hiện là một research snapshot của run ticker ví dụ. Các metric holdout được giữ nguyên để minh họa quality gate, training có thể tái lập, inference hybrid, kiểm tra checksum, drift check và cơ chế governance; việc promotion cho ticker mục tiêu cần một run riêng.

## Giới hạn

Run hiện tại chỉ bao phủ một ticker và technical feature từ OHLCV. Đây là giới hạn của dữ liệu minh hoạ, không phải phạm vi thiết kế framework; cần đánh giá cross-ticker, cross-market và regime shift trước khi promote.
