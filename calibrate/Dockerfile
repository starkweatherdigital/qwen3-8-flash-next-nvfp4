# BCN-84: Flash-Next FP8-periphery activation calibration (rented B200)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir --break-system-packages -U transformers accelerate "huggingface_hub[hf_transfer]" pillow
ENV HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
COPY corpus/ /app/corpus/
COPY calib_server.py /app/calib_server.py
EXPOSE 8000
CMD ["python", "/app/calib_server.py"]
