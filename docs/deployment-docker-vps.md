# Panduan Deployment DIR-RAG Backend via Docker di VPS

> Panduan ini menjelaskan langkah demi langkah cara melakukan deployment backend **Humbet AI Chatbot** (FastAPI + RAG) menggunakan **Docker** di VPS seperti Hostinger, Contabo, DigitalOcean, atau penyedia VPS lainnya.

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Akses VPS via SSH](#2-akses-vps-via-ssh)
3. [Instalasi Docker & Docker Compose](#3-instalasi-docker--docker-compose)
4. [Upload Kode ke VPS](#4-upload-kode-ke-vps)
5. [Konfigurasi Environment Variables](#5-konfigurasi-environment-variables)
6. [Build & Jalankan Container](#6-build--jalankan-container)
7. [Verifikasi Deployment](#7-verifikasi-deployment)
8. [Setup Nginx Reverse Proxy + SSL](#8-setup-nginx-reverse-proxy--ssl)
9. [Manajemen & Maintenance](#9-manajemen--maintenance)
10. [Troubleshooting](#10-troubleshooting)
11. [CI/CD Otomatis (Opsional)](#11-cicd-otomatis-opsional)

---

## 1. Prasyarat

### Spesifikasi VPS Minimum

| Komponen | Minimum       | Rekomendasi    |
|----------|---------------|----------------|
| CPU      | 1 vCPU        | 2 vCPU         |
| RAM      | 2 GB          | 4 GB           |
| Storage  | 20 GB SSD     | 40 GB SSD      |
| OS       | Ubuntu 22.04+ | Ubuntu 24.04   |
| Bandwidth| Unmetered     | Unmetered      |

> **Catatan:** Model embedding BGE dan reranker akan mengunduh model ke dalam container saat pertama kali jalan. Pastikan VPS memiliki storage cukup (~5 GB ekstra untuk model cache).

### Akun & Layanan yang Dibutuhkan

- [x] VPS aktif (misalnya Hostinger VPS KVM atau cloud VPS lainnya)
- [x] Domain (opsional, untuk SSL/HTTPS)
- [x] API key: **Google API Key** (Gemini), **OpenAI API Key** (opsional)
- [x] Database PostgreSQL (Supabase atau self-hosted)

---

## 2. Akses VPS via SSH

### Dari Windows (PowerShell / Terminal)

```powershell
ssh root@<IP_VPS_ANDA>
```

### Dari Windows (PuTTY)

1. Buka PuTTY.
2. Masukkan **Host Name**: `<IP_VPS_ANDA>`, **Port**: `22`.
3. Klik **Open**, login sebagai `root`.

### Hostinger Khusus

1. Login ke [hPanel Hostinger](https://hpanel.hostinger.com).
2. Buka menu **VPS** → pilih VPS Anda.
3. Catat **IP Address** dan **Root Password** dari tab **Overview**.
4. Gunakan SSH atau **Browser Terminal** bawaan Hostinger.

> **Tips Keamanan:** Setelah berhasil login, segera buat user non-root:
>
> ```bash
> adduser deploy
> usermod -aG sudo deploy
> ```

---

## 3. Instalasi Docker & Docker Compose

Jalankan perintah berikut satu per satu di VPS:

### 3.1 Update Sistem

```bash
sudo apt update && sudo apt upgrade -y
```

### 3.2 Instal Dependensi

```bash
sudo apt install -y ca-certificates curl gnupg lsb-release
```

### 3.3 Tambahkan Docker GPG Key & Repository

```bash
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 3.4 Instal Docker Engine & Compose Plugin

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3.5 Verifikasi Instalasi

```bash
docker --version
docker compose version
```

### 3.6 (Opsional) Jalankan Docker Tanpa `sudo`

```bash
sudo usermod -aG docker $USER
# Logout dan login kembali agar perubahan berlaku
exit
```

---

## 4. Upload Kode ke VPS

### Opsi A: Clone dari GitHub (Rekomendasi)

```bash
cd /opt
sudo git clone https://github.com/<USERNAME>/DIR-RAG-Backend.git
cd DIR-RAG-Backend
```

### Opsi B: Upload Manual via SCP

Dari **komputer lokal** (PowerShell):

```powershell
# Compress folder terlebih dahulu
Compress-Archive -Path ".\DIR-RAG-Backend\*" -DestinationPath ".\dir-rag-backend.zip"

# Upload ke VPS
scp .\dir-rag-backend.zip root@<IP_VPS>:/opt/
```

Kemudian di **VPS**:

```bash
cd /opt
sudo apt install -y unzip
unzip dir-rag-backend.zip -d DIR-RAG-Backend
cd DIR-RAG-Backend
```

### Opsi C: Upload via SFTP (FileZilla)

1. Buka FileZilla → **File → Site Manager**.
2. **Protocol**: SFTP, **Host**: `<IP_VPS>`, **Port**: 22.
3. **User**: `root`, **Password**: password VPS.
4. Upload folder project ke `/opt/DIR-RAG-Backend`.

---

## 5. Konfigurasi Environment Variables

### 5.1 Buat File `.env`

```bash
cd /opt/DIR-RAG-Backend
nano .env
```

### 5.2 Isi `.env` dengan Konfigurasi Berikut

```env
# ===== LLM & Embedding =====
GOOGLE_API_KEY=<google_api_key_anda>
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001

# OpenAI (opsional, jika menggunakan OpenAI)
# OPENAI_API_KEY=<openai_api_key_anda>
# GPT_MODEL=gpt-4
# EMBEDDING_MODEL=text-embedding-ada-002

# ===== RAG Configuration =====
RAG_MODE=modular
VECTOR_BACKEND=chroma
USE_BGE=False
SIMILARITY_TOP_K=5
ENABLE_DRAGIN=False
DRAGIN_LLM_BACKEND=gemini

# ===== Embedding Model (BGE) =====
BGE_MODEL_NAME=intfloat/multilingual-e5-base

# ===== Reranker =====
RERANKER_MODEL=amberoad/bert-multilingual-passage-reranking-msmarco

# ===== Database =====
DATABASE_URL=postgresql+asyncpg://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>

# ===== Authentication =====
JWT_SECRET_KEY=<buat_secret_key_random_64_char>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXP_HOURS=24
REFRESH_TOKEN_EXP_DAYS=7
DEFAULT_ADMIN_PASSWORD=<password_admin_anda>

# ===== Context & Memory =====
CONTEXT_MAX_DOCS=5
CONTEXT_CHAR_BUDGET=10000
MEMORY_MAX_TURNS=5
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT=95

# ===== Query Refinement =====
RQ_ENABLE_BYPASS=True

# ===== CORS (sesuaikan dengan domain frontend) =====
CORS_ALLOW_ORIGINS=["https://yourdomain.com","http://localhost:3000"]

# ===== Environment =====
ENVIRONMENT=prod
```

> **⚠️ Penting:**
> - Ganti semua placeholder `<...>` dengan nilai sebenarnya.
> - Untuk `JWT_SECRET_KEY`, generate random string:
>   ```bash
>   openssl rand -hex 32
>   ```
> - Jangan commit file `.env` ke Git (sudah di-exclude via `.gitignore`).

### 5.3 Amankan File `.env`

```bash
chmod 600 .env
```

---

## 6. Build & Jalankan Container

### 6.1 Review `docker-compose.yml`

File `docker-compose.yml` yang sudah ada di repo:

```yaml
version: "3.9"

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dir-rag-backend
    ports:
      - "8080:8080"
    env_file:
      - .env
    volumes:
      - ./storage:/app/storage
    restart: unless-stopped
```

> **Penjelasan:**
> - `ports: "8080:8080"` — meneruskan port 8080 dari container ke host.
> - `env_file: .env` — memuat semua variabel lingkungan dari file `.env`.
> - `volumes: ./storage:/app/storage` — menyimpan data indeks vektor dan log secara persisten di host.
> - `restart: unless-stopped` — container otomatis restart jika crash atau VPS reboot.

### 6.2 Build Image

```bash
cd /opt/DIR-RAG-Backend
docker compose build
```

> Proses build pertama kali membutuhkan waktu **5–15 menit** tergantung kecepatan internet VPS (mengunduh base image Python dan pip dependencies).

### 6.3 Jalankan Container

```bash
docker compose up -d
```

Flag `-d` menjalankan container di background (detached mode).

### 6.4 Cek Status Container

```bash
docker compose ps
```

Output yang diharapkan:

```
NAME               IMAGE                      STATUS          PORTS
dir-rag-backend    dir-rag-backend-backend    Up X minutes    0.0.0.0:8080->8080/tcp
```

### 6.5 Lihat Log Aplikasi

```bash
# Log real-time
docker compose logs -f backend

# Log 100 baris terakhir
docker compose logs --tail=100 backend
```

---

## 7. Verifikasi Deployment

### 7.1 Health Check

```bash
curl http://localhost:8080/health
```

Response yang diharapkan:

```json
{"status": "ok", "rag_mode": "modular"}
```

### 7.2 Test dari Luar VPS

Buka browser dan akses:

```
http://<IP_VPS>:8080/health
```

> **Catatan Hostinger:** Pastikan port 8080 dibuka di **firewall VPS**.
> Di Hostinger hPanel: **VPS → Firewall → Add Rule**:
> - Protocol: TCP
> - Port: 8080
> - Source: Anywhere (0.0.0.0/0)

### 7.3 Test Endpoint Chat

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa itu jurusan informatika?"}'
```

---

## 8. Setup Nginx Reverse Proxy + SSL

Agar backend bisa diakses via HTTPS dengan domain (misalnya `api.yourdomain.com`).

### 8.1 Instal Nginx & Certbot

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### 8.2 Konfigurasi Nginx

```bash
sudo nano /etc/nginx/sites-available/dir-rag-backend
```

Isi dengan:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;    # Ganti dengan domain Anda

    # Redirect HTTP ke HTTPS (akan aktif setelah SSL terpasang)
    # return 301 https://$server_name$request_uri;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout lebih panjang untuk RAG processing
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;

        # Untuk request body besar (upload dokumen)
        client_max_body_size 50M;
    }
}
```

### 8.3 Aktifkan Konfigurasi

```bash
sudo ln -s /etc/nginx/sites-available/dir-rag-backend /etc/nginx/sites-enabled/
sudo nginx -t          # Test konfigurasi, harus output "OK"
sudo systemctl reload nginx
```

### 8.4 Pasang SSL dengan Let's Encrypt

```bash
sudo certbot --nginx -d api.yourdomain.com
```

Ikuti prompt yang muncul:
1. Masukkan email.
2. Setujui Terms of Service.
3. Pilih **Redirect HTTP to HTTPS** (opsi 2).

Certbot akan otomatis:
- Mengunduh sertifikat SSL.
- Mengedit konfigurasi Nginx untuk HTTPS.
- Mengatur auto-renewal.

### 8.5 Verifikasi HTTPS

```bash
curl https://api.yourdomain.com/health
```

### 8.6 (Opsional) Tutup Port 8080 dari Akses Publik

Setelah Nginx terpasang, sebaiknya port 8080 hanya diakses dari localhost:

```bash
sudo ufw allow 22      # SSH
sudo ufw allow 80      # HTTP
sudo ufw allow 443     # HTTPS
sudo ufw deny 8080     # Block akses langsung ke backend
sudo ufw enable
```

---

## 9. Manajemen & Maintenance

### 9.1 Perintah Docker Compose Umum

| Perintah                           | Fungsi                                  |
|------------------------------------|-----------------------------------------|
| `docker compose up -d`             | Jalankan container di background        |
| `docker compose down`              | Stop & hapus container                  |
| `docker compose restart`           | Restart container                       |
| `docker compose logs -f backend`   | Lihat log real-time                     |
| `docker compose build --no-cache`  | Rebuild image tanpa cache               |
| `docker compose pull`              | Pull image terbaru (jika pakai registry)|

### 9.2 Update Aplikasi

```bash
cd /opt/DIR-RAG-Backend

# Jika menggunakan Git
git pull origin main

# Rebuild dan restart
docker compose down
docker compose build
docker compose up -d
```

### 9.3 Backup Data

```bash
# Backup storage (vector index + logs)
tar -czf backup-storage-$(date +%Y%m%d).tar.gz storage/

# Backup .env
cp .env .env.backup
```

### 9.4 Auto-Restart saat VPS Reboot

Sudah dihandle oleh konfigurasi `restart: unless-stopped` di `docker-compose.yml`. Pastikan Docker service juga aktif saat boot:

```bash
sudo systemctl enable docker
```

### 9.5 Monitoring Disk Space

```bash
# Cek penggunaan disk
df -h

# Cek ukuran Docker images & containers
docker system df

# Bersihkan resource Docker yang tidak terpakai
docker system prune -a --volumes
```

> ⚠️ Hati-hati dengan `docker system prune -a`, perintah ini menghapus semua image yang tidak digunakan.

---

## 10. Troubleshooting

### Container Tidak Mau Start

```bash
# Cek log error
docker compose logs backend

# Cek apakah port sudah dipakai
sudo lsof -i :8080
```

**Penyebab umum:**
- Port 8080 sudah digunakan proses lain → ubah port mapping di `docker-compose.yml`.
- File `.env` tidak ditemukan atau format salah → cek path dan syntax.
- Database tidak bisa diakses → pastikan `DATABASE_URL` benar dan VPS bisa akses host database.

### Error: Database Connection Refused

```bash
# Test koneksi ke database dari VPS
sudo apt install -y postgresql-client
psql "postgresql://<USER>:<PASS>@<HOST>:<PORT>/<DB>"
```

Jika pakai Supabase, pastikan:
- IP VPS tidak di-block oleh Supabase.
- Gunakan port `5432` (direct connection) atau `6543` (pooler/transaction mode).

### Error: Permission Denied pada `/app/storage`

```bash
# Perbaiki ownership
sudo chown -R 1001:1001 ./storage
```

Angka `1001` sesuai dengan UID user `appuser` di Dockerfile.

### Container Kehabisan Memory (OOM Killed)

```bash
# Cek apakah container di-kill karena OOM
docker inspect dir-rag-backend | grep -i oom

# Tambahkan memory limit di docker-compose.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 3G
```

### SSL Certificate Tidak Ter-renew

```bash
# Test renewal
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal
```

---

## 11. CI/CD Otomatis (Opsional)

### Menggunakan GitHub Actions

Buat file `.github/workflows/deploy.yml` di repository:

```yaml
name: Deploy to VPS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            cd /opt/DIR-RAG-Backend
            git pull origin main
            docker compose down
            docker compose build
            docker compose up -d
            echo "Deployment selesai pada $(date)"
```

**Setup GitHub Secrets:**

Di GitHub repo → **Settings → Secrets and variables → Actions**, tambahkan:

| Secret Name    | Value                                |
|----------------|--------------------------------------|
| `VPS_HOST`     | IP address VPS                       |
| `VPS_USER`     | `root` atau user deploy              |
| `VPS_SSH_KEY`  | Private SSH key (isi file `id_rsa`)  |

**Generate SSH Key (di VPS):**

```bash
ssh-keygen -t ed25519 -C "github-actions"
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/id_ed25519   # Copy ini ke GitHub Secret VPS_SSH_KEY
```

---

## Ringkasan Arsitektur Deployment

```
┌─────────────────────────────────────────────────────────┐
│                        VPS (Ubuntu)                     │
│                                                         │
│  ┌──────────┐      ┌──────────────────────────────┐     │
│  │  Nginx   │──────│  Docker Container             │     │
│  │ :80/:443 │      │  ┌──────────────────────────┐ │     │
│  │ (SSL)    │      │  │  FastAPI (uvicorn :8080)  │ │     │
│  └──────────┘      │  │                          │ │     │
│                    │  │  - RAG Pipeline           │ │     │
│                    │  │  - Vector Store (FAISS)   │ │     │
│                    │  │  - Auth (JWT)             │ │     │
│                    │  └──────────────────────────┘ │     │
│                    │                              │     │
│                    │  Volume: ./storage            │     │
│                    └──────────────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          │ DATABASE_URL
                          ▼
                ┌──────────────────┐
                │  PostgreSQL      │
                │  (Supabase /     │
                │   Self-hosted)   │
                └──────────────────┘
```

---

## Referensi Cepat

```bash
# === Deployment Pertama Kali ===
cd /opt/DIR-RAG-Backend
nano .env                          # Konfigurasi environment
docker compose build               # Build image
docker compose up -d               # Jalankan
curl http://localhost:8080/health   # Verifikasi

# === Update ===
git pull origin main
docker compose down
docker compose build
docker compose up -d

# === Monitoring ===
docker compose ps                  # Status container
docker compose logs -f backend     # Log real-time
docker stats                       # Resource usage
```

---

*Dokumen ini dibuat untuk keperluan deployment skripsi. Terakhir diperbarui: April 2026.*
