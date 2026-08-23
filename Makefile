.PHONY: help install data validate notebooks train smoke predict monitor api test frontend lint format clean

PYTHON := python

help:
	@echo "data       Tải dữ liệu OHLCV của VIC.VN"
	@echo "notebooks  Chạy notebook 01-05 theo đúng thứ tự"
	@echo "train      Train và đóng gói GRU candidate đã khóa"
	@echo "predict    Chạy batch inference trên CSV đã cấu hình"
	@echo "monitor    Tính feature drift gần nhất"
	@echo "test       Chạy bộ test Python"

install:
	pip install -e ".[dev,api,notebooks]"

data:
	$(PYTHON) scripts/download_data.py --ticker VIC

validate:
	$(PYTHON) scripts/validate_data.py

notebooks:
	$(PYTHON) scripts/run_notebooks.py

train:
	$(PYTHON) scripts/train.py

smoke:
	$(PYTHON) scripts/train.py --smoke-test

predict:
	$(PYTHON) scripts/batch_predict.py

monitor:
	$(PYTHON) scripts/monitor.py

api:
	$(PYTHON) -m uvicorn backend.main:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov=backend --cov-report=term-missing

frontend:
	npm run build --prefix frontend

lint:
	flake8 src/ scripts/ tests/ backend/ --max-line-length=100 --extend-ignore=E203,W503,E402,E501

format:
	isort src/ scripts/ tests/ backend/
	black src/ scripts/ tests/ backend/

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
