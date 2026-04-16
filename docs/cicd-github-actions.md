# Panduan CI/CD dengan GitHub Actions – Auto Deploy ke VPS

> Panduan ini menjelaskan cara mengatur **Continuous Deployment (CD)** menggunakan GitHub Actions sehingga setiap kali ada push ke branch `main`, backend **DIR-RAG** akan otomatis di-redeploy ke VPS.

---

## Daftar Isi

1. [Gambaran Alur CI/CD](#1-gambaran-alur-cicd)
2. [Prasyarat](#2-prasyarat)
3. [Persiapan di VPS](#3-persiapan-di-vps)
4. [Persiapan di GitHub](#4-persiapan-di-github)
5. [Membuat Workflow GitHub Actions](#5-membuat-workflow-github-actions)
6. [Penjelasan Workflow](#6-penjelasan-workflow)
7. [Testing Pipeline](#7-testing-pipeline)
8. [Monitoring & Notifikasi](#8-monitoring--notifikasi)
9. [Rollback jika Gagal](#9-rollback-jika-gagal)
10. [Tips & Best Practices](#10-tips--best-practices)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Gambaran Alur CI/CD

```
┌──────────────┐     push ke main     ┌──────────────────┐
│  Developer   │ ──────────────────── │  GitHub Repo     │
│  (Lokal)     │                      │  branch: main    │
└──────────────┘                      └────────┬─────────┘
                                               │
                                               │ trigger
                                               ▼
                                      ┌──────────────────┐
                                      │  GitHub Actions   │
                                      │  Runner           │
                                      │                  │
                                      │  1. Checkout code │
                                      │  2. SSH ke VPS    │
                                      │  3. git pull      │
                                      │  4. docker build  │
                                      │  5. docker up     │
                                      │  6. health check  │
                                      └────────┬─────────┘
                                               │
                                               │ SSH
                                               ▼
                                      ┌──────────────────┐
                                      │  VPS (Hostinger)  │
                                      │                  │
                                      │  Docker Container │
                                      │  dir-rag-backend  │
                                      └──────────────────┘
```

**Alur Singkat:**

1. Developer push perubahan kode ke branch `main`.
2. GitHub Actions secara otomatis berjalan.
3. Runner terhubung ke VPS via SSH.
4. Di VPS: `git pull` → `docker compose build` → `docker compose up -d`.
5. Health check memastikan container berjalan normal.
6. Jika gagal, otomatis rollback ke versi sebelumnya.

---

## 2. Prasyarat

Pastikan hal-hal berikut sudah terpenuhi:

- [x] Backend sudah ter-deploy manual di VPS (lihat `deployment-docker-vps.md`)
- [x] Repository sudah di-push ke GitHub
- [x] Docker & Docker Compose sudah terinstal di VPS
- [x] Git sudah terinstal di VPS
- [x] VPS bisa diakses via SSH

---

## 3. Persiapan di VPS

### 3.1 Buat SSH Key Khusus untuk GitHub Actions

Login ke VPS, lalu buat SSH key baru yang khusus digunakan oleh GitHub Actions:

```bash
# Login ke VPS
ssh root@<IP_VPS>

# Buat SSH key baru (tanpa passphrase, tekan Enter 2x)
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_actions_key
```

Output akan menghasilkan dua file:

| File                              | Fungsi                                  |
|-----------------------------------|-----------------------------------------|
| `~/.ssh/github_actions_key`       | **Private key** → simpan di GitHub      |
| `~/.ssh/github_actions_key.pub`   | **Public key** → simpan di VPS          |

### 3.2 Tambahkan Public Key ke `authorized_keys`

```bash
cat ~/.ssh/github_actions_key.pub >> ~/.ssh/authorized_keys
```

### 3.3 Salin Private Key (untuk Disimpan di GitHub Nanti)

```bash
cat ~/.ssh/github_actions_key
```

> **⚠️ Salin seluruh output**, termasuk baris `-----BEGIN OPENSSH PRIVATE KEY-----` dan `-----END OPENSSH PRIVATE KEY-----`. Simpan sementara di notepad.

### 3.4 Pastikan Permission SSH Benar

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/github_actions_key
```

### 3.5 Pastikan Repo Sudah Di-clone di VPS

```bash
# Jika belum ada
cd /opt
git clone https://github.com/<USERNAME>/DIR-RAG-Backend.git
cd DIR-RAG-Backend

# Jika sudah ada, pastikan remote origin benar
cd /opt/DIR-RAG-Backend
git remote -v
# Harus menunjukkan URL GitHub Anda
```

### 3.6 (Opsional) Setup Deploy User

Untuk keamanan lebih baik, buat user khusus deploy:

```bash
# Buat user
adduser deploy
usermod -aG docker deploy
usermod -aG sudo deploy

# Pindahkan SSH key ke user deploy
mkdir -p /home/deploy/.ssh
cp ~/.ssh/github_actions_key.pub /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

# Berikan akses ke folder project
chown -R deploy:deploy /opt/DIR-RAG-Backend
```

---

## 4. Persiapan di GitHub

### 4.1 Buka Repository Settings

1. Buka repository di GitHub: `https://github.com/<USERNAME>/DIR-RAG-Backend`
2. Klik tab **Settings** (ikon gear ⚙️)
3. Di sidebar kiri, klik **Secrets and variables → Actions**

### 4.2 Tambahkan Repository Secrets

Klik **New repository secret** untuk setiap variabel berikut:

| Secret Name       | Value                                       | Keterangan                                  |
|-------------------|---------------------------------------------|---------------------------------------------|
| `VPS_HOST`        | `<IP_VPS_ANDA>`                             | Contoh: `103.xxx.xxx.xxx`                   |
| `VPS_USER`        | `root` atau `deploy`                        | User SSH di VPS                             |
| `VPS_SSH_KEY`     | *(isi private key dari langkah 3.3)*        | Seluruh isi file `github_actions_key`       |
| `VPS_PORT`        | `22`                                        | Port SSH (default 22)                       |
| `DEPLOY_PATH`     | `/opt/DIR-RAG-Backend`                      | Path project di VPS                         |

**Cara menambahkan secret:**

1. Klik **New repository secret**
2. **Name**: masukkan nama secret (contoh: `VPS_HOST`)
3. **Secret**: masukkan value-nya
4. Klik **Add secret**

Ulangi untuk semua 5 secret di atas.

> **Tampilan akhir di GitHub:**
> ```
> Repository secrets (5)
> ├── DEPLOY_PATH
> ├── VPS_HOST
> ├── VPS_PORT
> ├── VPS_SSH_KEY
> └── VPS_USER
> ```

---

## 5. Membuat Workflow GitHub Actions

### 5.1 Buat Folder Workflow

Di komputer lokal, buat folder dan file berikut:

```
DIR-RAG-Backend/
└── .github/
    └── workflows/
        └── deploy.yml
```

### 5.2 Isi File `deploy.yml`

Buat file `.github/workflows/deploy.yml` dengan isi berikut:

```yaml
# ============================================================
# CI/CD Pipeline - Auto Deploy DIR-RAG Backend ke VPS
# ============================================================
# Trigger: Push ke branch 'main'
# Alur: SSH ke VPS → git pull → docker build → docker up → health check
# ============================================================

name: 🚀 Deploy to VPS

on:
  push:
    branches:
      - main
    # Hanya trigger jika file-file berikut berubah
    # (hapus bagian 'paths' jika ingin trigger untuk semua perubahan)
    paths:
      - 'app/**'
      - 'Dockerfile'
      - 'docker-compose.yml'
      - 'requirements.txt'

  # Memungkinkan trigger manual dari GitHub UI
  workflow_dispatch:

# Hanya izinkan 1 deployment berjalan pada satu waktu
concurrency:
  group: deploy-production
  cancel-in-progress: false

jobs:
  deploy:
    name: 🖥️ Deploy ke VPS
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      # ── Step 1: Tampilkan info deployment ──
      - name: 📋 Info Deployment
        run: |
          echo "🚀 Deployment dipicu oleh: ${{ github.actor }}"
          echo "📦 Commit: ${{ github.sha }}"
          echo "📝 Pesan: ${{ github.event.head_commit.message }}"
          echo "🕐 Waktu: $(date +'%Y-%m-%d %H:%M:%S UTC')"

      # ── Step 2: Deploy via SSH ──
      - name: 🔑 Deploy ke VPS via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          port: ${{ secrets.VPS_PORT }}
          timeout: 300s
          command_timeout: 600s
          script: |
            set -e

            echo "=========================================="
            echo "🚀 Memulai deployment..."
            echo "=========================================="

            # Masuk ke direktori project
            cd ${{ secrets.DEPLOY_PATH }}

            # ── Simpan commit saat ini untuk rollback ──
            PREVIOUS_COMMIT=$(git rev-parse HEAD)
            echo "📌 Commit sebelumnya: $PREVIOUS_COMMIT"

            # ── Pull perubahan terbaru ──
            echo ""
            echo "📥 Pull perubahan terbaru dari GitHub..."
            git fetch origin main
            git reset --hard origin/main

            CURRENT_COMMIT=$(git rev-parse HEAD)
            echo "📌 Commit terbaru: $CURRENT_COMMIT"

            # ── Cek apakah ada perubahan ──
            if [ "$PREVIOUS_COMMIT" = "$CURRENT_COMMIT" ]; then
              echo "✅ Tidak ada perubahan. Deployment dilewati."
              exit 0
            fi

            # ── Build ulang Docker image ──
            echo ""
            echo "🔨 Building Docker image..."
            docker compose build --no-cache

            # ── Stop container lama ──
            echo ""
            echo "⏹️  Menghentikan container lama..."
            docker compose down

            # ── Jalankan container baru ──
            echo ""
            echo "▶️  Menjalankan container baru..."
            docker compose up -d

            # ── Tunggu container ready ──
            echo ""
            echo "⏳ Menunggu container ready..."
            sleep 15

            # ── Health check ──
            echo ""
            echo "🏥 Health check..."
            RETRY=0
            MAX_RETRY=5

            until curl -sf http://localhost:8080/health > /dev/null 2>&1; do
              RETRY=$((RETRY + 1))
              if [ $RETRY -ge $MAX_RETRY ]; then
                echo "❌ Health check gagal setelah $MAX_RETRY percobaan!"
                echo ""
                echo "🔄 Rollback ke commit sebelumnya..."
                git reset --hard $PREVIOUS_COMMIT
                docker compose build --no-cache
                docker compose down
                docker compose up -d
                echo "⚠️  Rollback selesai. Deployment GAGAL."
                exit 1
              fi
              echo "   Percobaan $RETRY/$MAX_RETRY - Menunggu 10 detik..."
              sleep 10
            done

            HEALTH_RESPONSE=$(curl -s http://localhost:8080/health)
            echo "✅ Health check berhasil: $HEALTH_RESPONSE"

            # ── Bersihkan Docker resources lama ──
            echo ""
            echo "🧹 Membersihkan image lama..."
            docker image prune -f

            echo ""
            echo "=========================================="
            echo "✅ Deployment BERHASIL!"
            echo "   Commit: $CURRENT_COMMIT"
            echo "   Waktu : $(date +'%Y-%m-%d %H:%M:%S')"
            echo "=========================================="

      # ── Step 3: Tampilkan hasil ──
      - name: ✅ Deployment Selesai
        if: success()
        run: |
          echo "🎉 Deployment ke VPS berhasil!"
          echo "📦 Commit: ${{ github.sha }}"

      - name: ❌ Deployment Gagal
        if: failure()
        run: |
          echo "⚠️ Deployment GAGAL!"
          echo "📦 Commit: ${{ github.sha }}"
          echo "Silakan cek log di atas untuk detail error."
```

---

## 6. Penjelasan Workflow

### Trigger

```yaml
on:
  push:
    branches:
      - main
    paths:
      - 'app/**'
      - 'Dockerfile'
      - 'docker-compose.yml'
      - 'requirements.txt'
  workflow_dispatch:
```

| Trigger | Kapan aktif |
|---------|-------------|
| `push → main` | Setiap push ke branch `main` |
| `paths` | Hanya jika file di folder `app/`, `Dockerfile`, `docker-compose.yml`, atau `requirements.txt` berubah |
| `workflow_dispatch` | Bisa di-trigger manual dari GitHub UI |

> **💡 Tips:** Jika ingin trigger untuk **semua perubahan** (termasuk update docs, README, dll), hapus bagian `paths`.

### Concurrency

```yaml
concurrency:
  group: deploy-production
  cancel-in-progress: false
```

Memastikan hanya **satu deployment berjalan** pada satu waktu. Jika ada push baru saat deployment sedang berjalan, push baru akan **menunggu** (bukan membatalkan).

### Health Check & Rollback Otomatis

Workflow melakukan health check setelah container dijalankan:

1. Coba akses `http://localhost:8080/health` hingga 5 kali.
2. Jika gagal → otomatis rollback ke commit sebelumnya.
3. Jika sukses → deployment berhasil.

---

## 7. Testing Pipeline

### 7.1 Push Workflow ke GitHub

Di komputer lokal:

```powershell
# Pastikan berada di root project
cd DIR-RAG-Backend

# Buat folder workflow
mkdir -p .github/workflows

# (Buat file deploy.yml seperti di atas)

# Commit dan push
git add .github/workflows/deploy.yml
git commit -m "ci: tambah GitHub Actions auto-deploy ke VPS"
git push origin main
```

### 7.2 Pantau Workflow

1. Buka repository di GitHub.
2. Klik tab **Actions**.
3. Akan muncul workflow **"🚀 Deploy to VPS"** yang sedang berjalan.
4. Klik untuk melihat detail setiap step.

### 7.3 Trigger Manual

1. Buka tab **Actions** di GitHub.
2. Klik workflow **"🚀 Deploy to VPS"** di sidebar kiri.
3. Klik tombol **Run workflow** → pilih branch `main` → klik **Run workflow**.

### 7.4 Test dengan Perubahan Kecil

Buat perubahan kecil untuk memastikan pipeline berjalan:

```powershell
# Misalnya, tambahkan komentar di main.py
# Lalu push
git add -A
git commit -m "test: verifikasi CI/CD pipeline"
git push origin main
```

Buka tab **Actions** untuk memantau deployment otomatis.

---

## 8. Monitoring & Notifikasi

### 8.1 Badge Status di README

Tambahkan badge di `README.md` untuk menampilkan status deployment:

```markdown
![Deploy Status](https://github.com/<USERNAME>/DIR-RAG-Backend/actions/workflows/deploy.yml/badge.svg)
```

Badge akan menampilkan:
- 🟢 **passing** — deployment terakhir berhasil
- 🔴 **failing** — deployment terakhir gagal

### 8.2 Notifikasi Email

GitHub secara default mengirim email notifikasi jika workflow gagal. Pastikan:

1. Buka **GitHub Settings** (profil) → **Notifications**.
2. Aktifkan **Actions** notifications.

### 8.3 (Opsional) Notifikasi Telegram

Tambahkan step berikut di akhir workflow untuk kirim notifikasi ke Telegram:

```yaml
      - name: 📨 Notifikasi Telegram
        if: always()
        uses: appleboy/telegram-action@master
        with:
          to: ${{ secrets.TELEGRAM_CHAT_ID }}
          token: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          message: |
            ${{ job.status == 'success' && '✅' || '❌' }} Deployment ${{ job.status }}
            📦 Commit: ${{ github.sha }}
            📝 Pesan: ${{ github.event.head_commit.message }}
            👤 Oleh: ${{ github.actor }}
```

Tambahkan secrets baru di GitHub:
- `TELEGRAM_BOT_TOKEN`: token dari [@BotFather](https://t.me/BotFather)
- `TELEGRAM_CHAT_ID`: ID chat/grup Anda (gunakan [@userinfobot](https://t.me/userinfobot))

---

## 9. Rollback jika Gagal

### 9.1 Rollback Otomatis (Sudah Termasuk di Workflow)

Workflow sudah memiliki mekanisme rollback otomatis:
- Jika health check gagal 5 kali, kode otomatis di-revert ke commit sebelumnya dan container di-rebuild.

### 9.2 Rollback Manual via SSH

Jika perlu rollback manual:

```bash
ssh root@<IP_VPS>
cd /opt/DIR-RAG-Backend

# Lihat riwayat commit
git log --oneline -10

# Rollback ke commit tertentu
git reset --hard <COMMIT_HASH>

# Rebuild dan restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 9.3 Rollback Manual via GitHub

1. Buka tab **Actions** di GitHub.
2. Cari workflow run yang **terakhir berhasil**.
3. Klik **Re-run all jobs** untuk deploy ulang versi tersebut.

Atau revert commit via Git:

```powershell
# Di komputer lokal
git revert HEAD
git push origin main
# Ini akan trigger deployment ulang dengan perubahan di-revert
```

---

## 10. Tips & Best Practices

### 🔒 Keamanan

- **Jangan pernah** menyimpan API key, password, atau secret langsung di file workflow.
- Gunakan **GitHub Secrets** untuk semua data sensitif.
- Gunakan **deploy user** (bukan root) dengan hak akses minimal.
- Pertimbangkan SSH key dengan waktu expire.

### 📦 Git & Branch Strategy

- Lakukan development di branch terpisah (`feature/xxx`, `fix/xxx`).
- Buat **Pull Request** ke `main` untuk review.
- Hanya merge ke `main` jika sudah siap deploy.

```
feature/new-rag-module ──PR──> main ──auto deploy──> VPS
```

### ⚡ Optimasi Build

Jika build terlalu lama, pertimbangkan:

1. **Docker layer caching**: Ubah `docker compose build --no-cache` menjadi `docker compose build` (gunakan cache).
2. **Hanya rebuild jika dependencies berubah**: Tambahkan pengecekan di script:

```bash
# Cek apakah requirements.txt berubah
if git diff $PREVIOUS_COMMIT $CURRENT_COMMIT -- requirements.txt | grep -q .; then
  echo "📦 requirements.txt berubah, build tanpa cache..."
  docker compose build --no-cache
else
  echo "📦 Menggunakan cache..."
  docker compose build
fi
```

### 📝 Commit Message Convention

Gunakan prefix commit yang jelas:

| Prefix | Contoh | Trigger Deploy? |
|--------|--------|-----------------|
| `feat:` | `feat: tambah endpoint baru` | ✅ Ya |
| `fix:` | `fix: perbaiki bug RAG loop` | ✅ Ya |
| `docs:` | `docs: update README` | ❌ Tidak* |
| `ci:` | `ci: update workflow` | ❌ Tidak* |
| `test:` | `test: tambah unit test` | ❌ Tidak* |

*\*Tidak trigger jika menggunakan filter `paths` di workflow.*

---

## 11. Troubleshooting

### ❌ Error: "Permission denied (publickey)"

**Penyebab:** SSH key tidak cocok atau belum ditambahkan.

**Solusi:**

```bash
# Di VPS, pastikan public key ada di authorized_keys
cat ~/.ssh/authorized_keys

# Pastikan permission benar
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Pastikan sshd mengizinkan key authentication
sudo grep "PubkeyAuthentication" /etc/ssh/sshd_config
# Harus: PubkeyAuthentication yes

# Restart SSH jika ada perubahan
sudo systemctl restart sshd
```

Di GitHub, pastikan **VPS_SSH_KEY** berisi **seluruh** private key (termasuk header dan footer).

### ❌ Error: "Connection timed out"

**Penyebab:** IP VPS salah, port SSH salah, atau firewall memblokir.

**Solusi:**

```bash
# Pastikan bisa SSH dari lokal terlebih dahulu
ssh -i <private_key> root@<IP_VPS> -p 22

# Di VPS, cek firewall
sudo ufw status
sudo ufw allow 22/tcp
```

### ❌ Error: "docker compose: command not found"

**Penyebab:** Docker Compose belum terinstal sebagai plugin.

**Solusi:**

```bash
# Instal Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Atau install versi standalone
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### ❌ Workflow Berjalan Tapi Tidak Ada Perubahan di VPS

**Penyebab:** `git pull` gagal karena ada conflict atau perubahan lokal di VPS.

**Solusi:**

Workflow sudah menggunakan `git reset --hard origin/main` untuk memastikan kode VPS selalu sama dengan GitHub.

Jika masih bermasalah, SSH ke VPS dan cek manual:

```bash
cd /opt/DIR-RAG-Backend
git status
git log --oneline -5
```

### ❌ Health Check Gagal & Rollback Terjadi

**Penyebab:** Aplikasi crash saat startup (error di kode baru).

**Solusi:**

```bash
# Cek log container
docker compose logs --tail=50 backend

# Cek apakah container running
docker compose ps
```

Perbaiki error di kode, commit, dan push ulang.

---

## Struktur File Akhir

Setelah mengikuti panduan ini, struktur project Anda menjadi:

```
DIR-RAG-Backend/
├── .github/
│   └── workflows/
│       └── deploy.yml          ← Workflow CI/CD
├── app/
│   ├── api/
│   ├── core/
│   ├── data/
│   ├── db/
│   ├── evaluation/
│   ├── ingestion/
│   ├── rag/
│   ├── schemas/
│   └── main.py
├── docs/
│   ├── deployment-docker-vps.md
│   └── cicd-github-actions.md  ← Panduan ini
├── storage/
├── .env                        ← TIDAK di-commit
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Checklist Setup CI/CD

Gunakan checklist ini untuk memastikan semua langkah sudah dilakukan:

- [ ] SSH key dibuat di VPS (`github_actions_key`)
- [ ] Public key ditambahkan ke `~/.ssh/authorized_keys`
- [ ] GitHub Secret `VPS_HOST` ditambahkan
- [ ] GitHub Secret `VPS_USER` ditambahkan
- [ ] GitHub Secret `VPS_SSH_KEY` ditambahkan
- [ ] GitHub Secret `VPS_PORT` ditambahkan
- [ ] GitHub Secret `DEPLOY_PATH` ditambahkan
- [ ] File `.github/workflows/deploy.yml` dibuat dan di-push
- [ ] Test push ke `main` → workflow berjalan ✅
- [ ] Health check di VPS berhasil ✅
- [ ] (Opsional) Badge status ditambahkan di README

---

*Dokumen ini dibuat untuk keperluan deployment skripsi. Terakhir diperbarui: April 2026.*
