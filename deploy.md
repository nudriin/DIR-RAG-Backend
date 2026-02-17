Berikut panduan langkah‑demi‑langkah dari buka Google Cloud Console sampai aplikasi Anda ter‑deploy di Cloud Run, memakai Dockerfile dan konfigurasi yang sudah ada di repo backend Anda.

---

**0. Gambaran singkat arsitektur**

- Backend FastAPI di‑container pakai `Dockerfile`.
- Deploy ke **Cloud Run** (region hemat biaya, misalnya `asia-southeast2`).
- Image disimpan di **Artifact Registry**.
- **Chroma**: disimpan di folder `/app/storage/vectors` yang di‑mount ke **Cloud Storage bucket** (persistent).
- **SQLite**:
  - Default: file `storage/chat_history.db` di dalam container.
  - Production (disarankan): ganti ke **Cloud SQL Postgres** lewat `DATABASE_URL` (sudah didukung di `app/db/engine.py`).

Di bawah ini saya pakai contoh:

- `PROJECT_ID`: `my-gcp-project`
- Region: `asia-southeast2`
- Service name: `dir-rag-backend`
- Repo image: `dir-rag-backend`

Silakan ganti dengan nilai Anda sendiri.

---

### 1. Masuk ke Google Cloud dan pilih project

1. Buka: https://console.cloud.google.com
2. Login dengan akun Google Anda.
3. Di bar atas kiri, klik **Project selector** → pilih project yang akan dipakai, atau klik **New Project** untuk membuat project baru.
4. Pastikan **Billing** sudah aktif untuk project ini (kalau belum, aktifkan dulu di menu Billing).

---

### 2. Aktifkan API yang dibutuhkan

Masih di Google Cloud Console:

1. Di kiri atas, klik menu ☰ → **APIs & Services → Library**.
2. Aktifkan API berikut (klik dan tekan **Enable** satu per satu):
   - **Cloud Run API**
   - **Cloud Build API**
   - **Artifact Registry API**
   - **Cloud Storage API**
   - **Secret Manager API** (untuk menyimpan API key, JWT secret, dll)
   - (Opsional) **Cloud SQL Admin API** jika nanti pakai Cloud SQL.

---

### 3. Buka Cloud Shell dan siapkan kode

1. Di kanan atas console, klik ikon **Cloud Shell** (ikon terminal `>_`).
2. Tunggu sampai terminal Cloud Shell siap.
3. Di Cloud Shell, clone atau upload kode backend:

   - Jika repo ada di GitHub:

     ```bash
     git clone https://github.com/USERNAME/REPO.git
     cd REPO/DIR-RAG-Backend
     ```

   - Jika repo hanya ada di laptop:
     - Zip folder backend Anda di lokal.
     - Di Cloud Shell, klik tombol **Upload file** (ikon panah ke atas), upload zip tersebut.
     - Ekstrak di Cloud Shell:

       ```bash
       unzip be-DIR-RAG-Backend.zip
       cd DIR-RAG-Backend
       ```

Pastikan sekarang Anda berada di root project backend yang berisi `Dockerfile`, `docker-compose.yml`, `cloudbuild.yaml`, `deploy_cloud_run.sh`, dsb.

---

### 4. Buat bucket Cloud Storage untuk Chroma

1. Di Cloud Shell:

   ```bash
   PROJECT_ID="my-gcp-project"       # ganti
   REGION="asia-southeast2"          # ganti jika mau
   gcloud config set project "$PROJECT_ID"
   ```

2. Buat bucket untuk vector store Chroma (nama harus unik global, jadi biasanya pakai project id di dalam namanya):

   ```bash
   BUCKET_NAME="${PROJECT_ID}-chroma-vectors"
   gsutil mb -l "$REGION" "gs://${BUCKET_NAME}"
   ```

Bucket ini akan digunakan oleh Cloud Run lewat konfigurasi `cloudrun-service.yaml` yang mem‑mount ke `/app/storage/vectors`.

---

### 5. Buat repository Artifact Registry untuk image Docker

1. Di Cloud Shell:

   ```bash
   gcloud artifacts repositories create dir-rag-backend \
     --repository-format=docker \
     --location="$REGION" \
     --description="DIR RAG backend images"
   ```

2. Jika sudah pernah dibuat sebelumnya, perintah ini akan gagal; itu normal. Anda bisa abaikan error *already exists*.

---

### 6. Siapkan environment & secrets (konsep)

Sebelum deploy, pikirkan env var penting (tidak perlu langsung di‑set sekarang, tapi siapkan):

- `OPENAI_API_KEY` / `GOOGLE_API_KEY`
- `JWT_SECRET_KEY`
- `DEFAULT_ADMIN_PASSWORD`
- (Opsional) `DATABASE_URL` untuk Cloud SQL Postgres, misalnya:

  ```text
  postgresql+asyncpg://USER:PASSWORD@/DBNAME?host=/cloudsql/INSTANCE_CONNECTION_NAME
  ```

Kita akan set env var ini nanti di menu Cloud Run (lebih aman jika sumbernya dari **Secret Manager**).

---

### 7. Build & deploy pertama kali ke Cloud Run (manual)

Kita pakai script yang sudah ada: `deploy_cloud_run.sh`.

1. Pastikan Anda di root project backend:

   ```bash
   ls
   # Harus terlihat: Dockerfile, deploy_cloud_run.sh, app/, dll
   ```

2. Jadikan script executable dan jalankan:

   ```bash
   chmod +x deploy_cloud_run.sh

   export PROJECT_ID="my-gcp-project"
   export REGION="asia-southeast2"

   ./deploy_cloud_run.sh
   ```

Script ini melakukan:

- `gcloud builds submit ...` → build image berdasarkan `Dockerfile` dan push ke Artifact Registry.
- `gcloud run deploy ...` → membuat service Cloud Run `dir-rag-backend` dengan:
  - `--min-instances=0` (hemat biaya, bisa 0 saat idle)
  - `--max-instances=3`
  - `--cpu=1`, `--memory=512Mi`
  - `--port=8080`
  - `--allow-unauthenticated` (bisa diakses publik, sesuaikan kebutuhan).

Tunggu sampai proses selesai; di output akan muncul **Service URL** Cloud Run.

Setelah langkah ini, service sudah jalan, tapi **belum mem‑mount bucket GCS untuk Chroma** dan belum ada env var rahasia. Langkah berikutnya menyempurnakan konfigurasi.

---

### 8. Pasang volume Cloud Storage ke Cloud Run (untuk Chroma)

1. Edit file `cloudrun-service.yaml` di repo Anda (bisa lewat editor di Cloud Shell atau di IDE lalu commit):

   - Ganti `PROJECT_ID` dengan project Anda.
   - Ganti `TAG` dengan tag image yang ingin dipakai, misalnya `latest`:

   ```yaml
   image: asia-southeast2-docker.pkg.dev/my-gcp-project/dir-rag-backend/dir-rag-backend:latest
   ...
   bucket: my-gcp-project-chroma-vectors
   ```

2. Terapkan konfigurasi ini ke Cloud Run:

   ```bash
   gcloud run services replace cloudrun-service.yaml \
     --project="$PROJECT_ID" \
     --region="$REGION"
   ```

Perintah ini akan meng‑update service Cloud Run:

- Volume `cloudStorage` bucket `PROJECT_ID-chroma-vectors` di‑mount ke `/app/storage/vectors`.
- Aplikasi tetap memakai port 8080.

Sekarang, data Chroma akan persisten di bucket ini.

---

### 9. Set environment variables & secrets di Cloud Run

1. Kembali ke **Google Cloud Console**.
2. Menu ☰ → **Cloud Run** → pilih service `dir-rag-backend`.
3. Klik tab **Variables & Secrets** / **Edit & Deploy New Revision**.
4. Di bagian **Environment variables**, tambahkan key penting:
   - `OPENAI_API_KEY` → pilih **Reference a secret** jika sudah buat di Secret Manager.
   - `GOOGLE_API_KEY` (jika pakai Gemini).
   - `JWT_SECRET_KEY`.
   - `DEFAULT_ADMIN_PASSWORD`.
   - (Opsional) `DATABASE_URL` jika pakai Cloud SQL.
5. Klik **Deploy** untuk membuat revision baru dengan env var tersebut.

Untuk membuat secret:

- Menu ☰ → **Security → Secret Manager** → **Create secret** → isi name & value.
- Nanti di Cloud Run, env var bisa di‑binding ke secret tersebut.

---

### 10. (Opsional) Integrasi database ke Cloud SQL

Jika Anda ingin SQLite diganti Cloud SQL Postgres:

1. Buat instance Cloud SQL Postgres via menu **SQL** di console.
2. Buat database dan user.
3. Catat:
   - `INSTANCE_CONNECTION_NAME` (format: `project:region:instance`)
   - Nama database, user, password.
4. Di Cloud Run:
   - Tambahkan annotation `run.googleapis.com/cloudsql-instances: INSTANCE_CONNECTION_NAME` (bisa via UI **Connections → Cloud SQL**).
   - Set `DATABASE_URL` di env var, misalnya:

     ```text
     postgresql+asyncpg://user:password@/dbname?host=/cloudsql/INSTANCE_CONNECTION_NAME
     ```

Aplikasi Anda akan otomatis pakai URL ini (lihat `DATABASE_URL` di `engine.py`).

---

### 11. Cek health dan endpoint

1. Di halaman service Cloud Run, klik URL service (format `https://dir-rag-backend-xxxxx-uc.a.run.app`).
2. Tambahkan `/health`:

   ```text
   https://.../health
   ```

3. Harus muncul JSON:

   ```json
   {"status":"ok","rag_mode": "..."}
   ```

Jika ini OK, backend sudah running.

---

### 12. (Opsional) Setup CI/CD otomatis dengan Cloud Build

Kalau ingin setiap push ke branch tertentu otomatis build & deploy:

1. Menu ☰ → **Cloud Build → Triggers** → **Create trigger**.
2. Pilih sumber repo (GitHub/Cloud Source) yang berisi project ini.
3. Atur:
   - Trigger event: `Push to a branch` (misalnya `main`).
   - Konfigurasi build: **cloudbuild.yaml** di root repo backend.
4. Simpan.

Setelah ini, setiap push ke branch tersebut akan menjalankan langkah yang didefinisikan di `cloudbuild.yaml` (build image + deploy ke Cloud Run dengan flags hemat biaya).

---

Kalau Anda mau, kirimkan:

- `PROJECT_ID` dan region yang Anda pilih
- Apakah mau pakai Cloud SQL atau cukup SQLite + GCS

Saya bisa bantu menuliskan contoh konkret isi `cloudrun-service.yaml` dan `DATABASE_URL` yang pas untuk konfigurasi Anda.