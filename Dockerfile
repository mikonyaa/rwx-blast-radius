FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
ADD --checksum=sha256:93b2445f15f395b81c20556777ece19a6c7507df950f13fa9fac4a3be739af0d https://codeload.github.com/tobymao/sqlglot/tar.gz/eb11aec4be8cb2122afe713a4816662c3f0fb2fa /tmp/sqlglot.tgz
RUN mkdir -p /opt/sqlglot && tar xzf /tmp/sqlglot.tgz --strip-components=1 -C /opt/sqlglot && rm /tmp/sqlglot.tgz && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/sqlglot SKIP_INTEGRATION=0 HOME=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY . /app
RUN python /app/query_probe.py cache /opt/sqlglot /app/query-cache
CMD ["python", "/app/selector.py"]
