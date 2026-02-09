.PHONY: sync sync-gpu help

help:
	@echo "Available targets:"
	@echo "  make sync      - Install CPU version (default)"
	@echo "  make sync-gpu  - Install GPU version with CUDA support"

sync:
	uv sync

sync-gpu:
	uv sync --extra gpu
	uv pip install \
		--index-url https://download.pytorch.org/whl/cu124 \
		--force-reinstall \
		--no-deps \
		"torch==2.6.0" \
		"torchvision==0.21.0" \
		"torchaudio==2.6.0"