PY ?= python
CONFIG ?= configs/ppo_fast.json
MODEL ?= runs/ppo_fast/best_model

.PHONY: help install selftest test scripted train eval play clean

help:
	@echo "make install    安装依赖"
	@echo "make selftest   自检(第一次必跑)"
	@echo "make test       纯逻辑单元测试(不需要 mujoco)"
	@echo "make scripted   跑脚本基线"
	@echo "make train      训练 PPO (CONFIG=configs/ppo_fast.json)"
	@echo "make eval       评测模型 (MODEL=runs/ppo_fast/best_model)"
	@echo "make play       可视化"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

selftest:
	$(PY) -m scripts.selftest

test:
	$(PY) tests/test_logic.py

scripted:
	$(PY) -m scripts.run_scripted

train:
	$(PY) -m scripts.train_ppo --config $(CONFIG)

eval:
	$(PY) -m scripts.evaluate --model $(MODEL)

play:
	$(PY) -m scripts.play --model $(MODEL)

clean:
	rm -rf runs __pycache__ .pytest_cache
