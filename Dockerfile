# Unified intent analyzer (all 5 agents, one checkpoint).
# On startup, entrypoint.py pulls the checkpoint from GCS
# ($ANALYZER_CHECKPOINT_GCS) into /srv/app/models, then serves.
FROM python:3.12-slim

WORKDIR /srv/app

# CPU torch: CUDA wheels are ~2.5GB and unnecessary for a ModernBERT
# decision model.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir \
      "transformers>=4.40" "scikit-learn>=1.3" "accelerate>=0.30" \
      "pyyaml>=6.0" "fastapi>=0.110" "uvicorn[standard]>=0.29" \
      "google-cloud-storage>=2.0" "httpx>=0.27"

COPY src/ ./src/
COPY policy/ ./policy/
COPY workflows/ ./workflows/
COPY studio/ ./studio/
COPY entrypoint.py .

EXPOSE 8080
ENV ANALYZER_CHECKPOINT_DIR=/srv/app/models \
    PORT=8080

CMD ["python", "entrypoint.py"]
