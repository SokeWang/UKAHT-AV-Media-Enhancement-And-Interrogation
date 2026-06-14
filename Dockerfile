# ==========================================
# Stage 1: Build the React + Vite Frontend
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy dependencies list and install
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

# Copy source code and build
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Build the FastAPI + Python Backend
# ==========================================
FROM python:3.10-slim-bullseye AS backend-runner

# Install system dependencies (including standard libraries for OpenCV/Image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend files
COPY backend/ ./backend/

# Copy compiled frontend from Stage 1 into the location FastAPI expects
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port 8000 and run the application
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
