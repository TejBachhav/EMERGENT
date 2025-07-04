# Stage 1: Build React App
FROM node:20.12.2-alpine3.20 AS frontend-build
# Use Alpine for smaller attack surface and latest Node version
WORKDIR /app

# Update Alpine packages to reduce vulnerabilities
RUN apk update && apk upgrade
COPY frontend/package.json frontend/yarn.lock /app/
# Split copying for better layer caching
RUN yarn install --frozen-lockfile --network-timeout 600000
COPY frontend/ /app/
ARG FRONTEND_ENV
ENV FRONTEND_ENV=${FRONTEND_ENV}
RUN rm -f /app/.env && \
    echo "${FRONTEND_ENV}" | tr ',' '\n' > /app/.env && \
    yarn build

# Stage 2: Install Python Backend
FROM python:3.12-alpine as backend
WORKDIR /app
# Install build dependencies for Python packages
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    python3-dev 
# Copy requirements first for better caching
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del .build-deps
# Copy application code after dependencies are installed
COPY backend/ /app/
# Remove any .env file to avoid leaking secrets
RUN rm -f /app/.env

# Stage 3: Final Image
FROM nginx:stable-alpine
# Create non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy built frontend
COPY --from=frontend-build /app/build /usr/share/nginx/html
# Copy backend
COPY --from=backend /app /backend
# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    # Security: fix permissions
    chown -R appuser:appgroup /backend /usr/share/nginx/html && \
    # Fix nginx permissions while allowing it to bind to port 80/443
    chown -R appuser:appgroup /var/cache/nginx /var/log/nginx /etc/nginx/conf.d && \
    chmod -R 755 /var/cache/nginx /var/log/nginx /etc/nginx/conf.d

# Install Python and dependencies with security in mind
# Pin versions for reproducibility and security
RUN apk add --no-cache python3~=3.11 py3-pip~=23.0 && \
    # Use separate layer to install dependencies to minimize image size
    pip3 install --break-system-packages --no-cache-dir -r /backend/requirements.txt

# Add env variables
ENV PYTHONUNBUFFERED=1
# Security: disable Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Security: enable Python hash randomization
ENV PYTHONHASHSEED=random

# Switch to non-root user for most operations
USER appuser

# Expose ports
EXPOSE 80

# Start both services: Flask and Nginx
CMD ["/entrypoint.sh"]
