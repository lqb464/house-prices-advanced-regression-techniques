"use client";

import {ChangeEvent, useEffect, useState} from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type ModelInfo = {
  model_version: string;
  model_type: string;
  ticker: string;
  trained_at: string;
  deployment_status: string;
  deployment_reasons: string[];
  holdout_metrics: {mae: number; rmse: number; directional_accuracy: number};
  zero_return_baseline_mae: number;
  checksum_verified?: boolean;
};

type Prediction = {
  as_of: string;
  prediction_percent: number;
  interval: {low: number; high: number; coverage_target: number};
  manual_review: boolean;
  input_warnings: string[];
  deployment_status: string;
};

const notebookSnapshot: ModelInfo = {
  model_version: "notebook-snapshot",
  model_type: "hybrid_ensemble",
  ticker: "VIC.VN",
  trained_at: "2026-08-26",
  deployment_status: "research_snapshot",
  deployment_reasons: [
    "Holdout MAE không tốt hơn zero-return baseline",
    "Interval coverage thấp hơn deployment floor",
  ],
  holdout_metrics: {mae: .0731303, rmse: .0971624, directional_accuracy: .403189},
  zero_return_baseline_mae: .0606715,
};

const experiments = [
  ["GRU · 20 phiên", 2.996, "RNN"],
  ["Hybrid ensemble · RNN + ML", 3.071, "Ensemble"],
  ["Zero return", 2.999, "Baseline"],
  ["Ridge", 3.092, "ML truyền thống"],
  ["Extra Trees", 3.118, "ML truyền thống"],
  ["HistGradientBoosting", 3.374, "ML truyền thống"],
  ["Return 5 ngày gần nhất", 4.065, "Baseline"],
] as const;

function parseCsv(text: string) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",").map(value => value.trim().toLowerCase());
  return lines.filter(Boolean).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]?.trim()]));
  });
}

export default function Home() {
  const [model, setModel] = useState<ModelInfo>(notebookSnapshot);
  const [apiState, setApiState] = useState("snapshot từ notebook");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [message, setMessage] = useState("Chọn CSV OHLCV của ticker để chạy raw-to-prediction.");

  useEffect(() => {
    fetch(`${API}/model`)
      .then(response => response.ok ? response.json() : Promise.reject())
      .then(data => { setModel(data); setApiState("artifact đang hoạt động"); })
      .catch(() => setApiState("snapshot từ notebook"));
  }, []);

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setMessage(`Đang validate ${file.name}…`);
    try {
      const records = parseCsv(await file.text());
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({records}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Dự báo thất bại");
      setPrediction(data);
      setMessage(`Đã xử lý ${records.length} dòng bằng ${data.model_version}.`);
    } catch (error) {
      setPrediction(null);
      setMessage(error instanceof Error ? error.message : "Không thể đọc file.");
    }
  }

  const relativeGap = (model.holdout_metrics.mae / model.zero_return_baseline_mae - 1) * 100;
  return <main>
    <header className="topbar">
      <a className="brand" href="#top"><span>SK</span><strong>StocKast</strong></a>
      <nav><a href="#experiments">Thí nghiệm</a><a href="#inference">Dự báo</a><a href="#governance">Quản trị</a></nav>
      <div className="status research_snapshot">chế độ nghiên cứu</div>
    </header>

    <section className="hero" id="top">
      <div>
        <p className="eyebrow">DATA SCIENTIST × ML ENGINEER · TIME SERIES</p>
        <h1>Dự báo có kiểm chứng,<br/><em>từ dữ liệu đến model.</em></h1>
        <p className="lede">Kết hợp ML truyền thống và RNN cho dự báo chuỗi thời gian cổ phiếu. Ticker VIC.VN trong màn hình này chỉ là ví dụ cấu hình.</p>
        <div className="heroActions"><a href="#experiments">Xem kết quả</a><span>API: {apiState}</span></div>
      </div>
      <aside className="gateCard">
        <p>ĐÁNH GIÁ HOLDOUT · RUN MINH HỌA</p><div className="gateIcon">✓</div><h2>Đã khóa để kiểm tra</h2>
        <strong>{(model.holdout_metrics.mae * 100).toFixed(2)}%</strong>
        <span>MAE · baseline chênh {relativeGap.toFixed(1)}%</span>
        <small>Run minh hoạ · 439 dự báo trên holdout</small>
      </aside>
    </section>

    <section className="metricStrip">
      <article><span>Target</span><strong>5 phiên</strong><small>Cumulative log-return</small></article>
      <article><span>Hybrid ensemble</span><strong>RNN + ML</strong><small>Component models kết hợp</small></article>
      <article><span>Đúng hướng trên holdout</span><strong>{(model.holdout_metrics.directional_accuracy * 100).toFixed(1)}%</strong><small>Không đạt 50%</small></article>
      <article><span>Interval coverage</span><strong>43.7%</strong><small>Mục tiêu 80%</small></article>
    </section>

    <section className="section" id="experiments">
      <div className="sectionTitle"><div><p>PHÒNG THÍ NGHIỆM MODEL</p><h2>Bảng xếp hạng trên validation</h2></div><span>Metric chính: MAE ↓</span></div>
      <div className="experimentGrid">
        <div className="leaderboard">
          {experiments.map(([name, mae, family], index) => <div className="modelRow" key={name}>
            <b>{String(index + 1).padStart(2, "0")}</b><div><strong>{name}</strong><small>{family}</small></div>
            <span className="bar"><i style={{width: `${Math.min(100, mae / 4.2 * 100)}%`}}/></span><em>{mae.toFixed(3)}%</em>
          </div>)}
        </div>
        <aside className="finding">
          <p>PHÁT HIỆN CHÍNH</p><h3>Component tốt nhất ≠ metric duy nhất.</h3>
          <p>Validation là nơi khóa component và tỷ trọng ensemble; holdout chỉ dùng một lần để báo cáo chất lượng của run.</p>
          <dl><div><dt>Validation MAE</dt><dd>2,996%</dd></div><div><dt>Holdout MAE</dt><dd>7,313%</dd></div><div><dt>Baseline holdout</dt><dd>6,067%</dd></div></dl>
        </aside>
      </div>
    </section>

    <section className="section split" id="inference">
      <div>
        <p className="eyebrow">DỮ LIỆU THÔ → FEATURES → COMPONENTS → ENSEMBLE</p><h2>Batch inference dùng cùng một pipeline.</h2>
        <p className="copy">CSV cần các cột date, open, high, low, close, volume và tối thiểu 70 phiên. Numeric string được coercion có cảnh báo; missing, sentinel và schema lỗi bị chặn.</p>
        <label className="upload">Chọn CSV ticker<input type="file" accept=".csv" onChange={upload}/></label>
        <p className="message">{message}</p>
      </div>
      <aside className="predictionCard">
        <p>KỊCH BẢN NGHIÊN CỨU MỚI NHẤT</p>
        {prediction ? <>
          <strong>{prediction.prediction_percent >= 0 ? "+" : ""}{prediction.prediction_percent.toFixed(2)}%</strong>
          <span>tại ngày {prediction.as_of} · horizon 5 phiên</span>
          <div><small>80% interval</small><b>{(prediction.interval.low * 100).toFixed(2)}% → {(prediction.interval.high * 100).toFixed(2)}%</b></div>
          <em>{prediction.manual_review ? "Cần rà soát thủ công" : "Không có cờ rà soát"}</em>
        </> : <><strong>—</strong><span>Chưa có batch inference</span><div><small>Phạm vi</small><b>artifact nghiên cứu</b></div></>}
      </aside>
    </section>

    <section className="section governance" id="governance">
      <div><p>QUẢN TRỊ MODEL</p><h2>Kết quả xấu không bị che đi.</h2></div>
      <ul>{model.deployment_reasons.map(reason => <li key={reason}>{reason}</li>)}<li>Không tự động đặt lệnh, phân bổ vốn hoặc đưa khuyến nghị đầu tư.</li><li>Monitor schema, feature drift, realized MAE và interval coverage trước mọi lần promote.</li></ul>
      <footer><span>Artifact checksum {model.checksum_verified ? "đã xác minh" : "có sau khi train local"}</span><span>Output notebook là nguồn metric chuẩn.</span></footer>
    </section>
  </main>;
}
