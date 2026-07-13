# 照片換臉 Web App

這是一個以 FastAPI、InsightFace 與 ONNX Runtime 建置的單機照片換臉服務。使用者可先建立可持久保存的人臉庫，再選擇來源臉與目標照片進行換臉；支援多臉選擇、結果下載、定時清理，以及 CPU／NVIDIA GPU 執行提供者自動選擇。

> 僅限使用本人或已取得明確同意的人像。禁止用於冒充、詐騙、未經同意的色情內容、身分驗證繞過或其他違法用途。

## 本機執行

需要 Python 3.12。先建立虛擬環境並安裝套件：

```bash
python -m venv .venv
```

Linux／macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export DATA_DIR="$PWD/data"
export MODEL_DIR="$PWD/models"
python scripts/download_models.py --model-dir "$MODEL_DIR"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:DATA_DIR = "$PWD\data"
$env:MODEL_DIR = "$PWD\models"
python scripts/download_models.py --model-dir $env:MODEL_DIR
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

開啟 `http://localhost:8080`。模型下載器會驗證既有檔案，已完整下載的模型不會重複下載。

## Docker 執行

Docker build 會執行模型下載器；若當下網路或模型來源不可用，建置仍會完成，容器啟動時會再重試。應用程式在模型缺失時仍可啟動，並透過介面與 `GET /api/health` 回報模型尚未就緒。

```bash
docker build -t faceswap-web .
docker run --rm -p 8080:8080 \
  -v faceswap-data:/data \
  -e PORT=8080 \
  faceswap-web
```

所有人臉、SQLite、暫存檔及結果都位於 `/data`；Docker volume 不可省略，否則刪除容器後資料會遺失。模型位於 `/models`，不應提交到 Git。

若需使用獲得授權的自訂模型來源，可在建置時傳入：

```bash
docker build \
  --build-arg INSWAPPER_MODEL_URL="<你的授權 inswapper 模型網址>" \
  --build-arg BUFFALO_L_MODEL_URL="<你的授權 buffalo_l 模型網址>" \
  -t faceswap-web .
```

執行中的容器也可用同名環境變數覆寫來源。只應使用自己有權存取與使用的網址。

## 上傳 GitHub

模型與 `/data` 已由 `.gitignore` 排除。建立自己的空白 GitHub repository 後執行：

```bash
git init
git add .
git commit -m "Initial face swap web app"
git branch -M main
git remote add origin https://github.com/你的帳號/你的專案.git
git push -u origin main
```

請勿以 Git LFS 或一般 Git commit 提交 ONNX 模型；部署時由下載腳本取得模型。

## Zeabur 部署

1. 將專案推送至 GitHub，在 Zeabur 建立服務並選擇該 repository。
2. 若此資料夾位於 monorepo，將服務的 Root Directory 設為 `faceswap-web`；若 repository 根目錄就是本專案則不需設定。
3. Zeabur 會偵測根目錄的 `Dockerfile`。建立網域後，服務會監聽 Zeabur 提供的 `PORT`。
4. 在服務的 **Volumes** 頁籤建立 Persistent Volume，將 **Mount Directory 精確設為 `/data`**，然後重新部署。
5. 保持單一服務副本。此專案使用 SQLite 與本機 volume，不適合讓多個副本同時寫入同一資料庫。

未掛載 `/data` 時，重新部署或重啟可能遺失人臉庫。掛載 volume 後的服務重啟會有短暫停機；重要資料仍應自行備份。

## 環境變數

| 變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `PORT` | `8080` | HTTP 監聽連接埠 |
| `DATA_DIR` | `/data` | 永久資料根目錄 |
| `MODEL_DIR` | `/models` | ONNX 模型目錄 |
| `MAX_UPLOAD_MB` | `15` | 單張上傳上限（MB） |
| `MAX_IMAGE_SIDE` | `2500` | 影像最大邊長，超過時等比例縮小 |
| `TEMP_RETENTION_HOURS` | `24` | 暫存圖保留時數 |
| `RESULT_RETENTION_HOURS` | `24` | 結果圖保留時數 |
| `MAX_CONCURRENT_JOBS` | `1` | 同時換臉工作數；CPU 建議維持 `1` |
| `FACE_DETECTION_SIZE` | `640` | 人臉偵測輸入尺寸 |
| `JPEG_QUALITY` | `95` | JPG 輸出品質 |
| `DATABASE_TIMEOUT_SECONDS` | `30` | SQLite 等待鎖定解除的秒數 |
| `CLEANUP_INTERVAL_SECONDS` | `3600` | 背景清理週期（秒） |
| `JOB_QUEUE_TIMEOUT_SECONDS` | `30` | 等待換臉工作空位的秒數 |
| `INSWAPPER_MODEL_URL` | 官方來源 | 授權模型下載網址覆寫 |
| `BUFFALO_L_MODEL_URL` | 官方來源 | 授權偵測模型包網址覆寫 |

## CPU 與 NVIDIA GPU

預設安裝 CPU 版 `onnxruntime`，不需要 GPU。若要建立 GPU 版：

```bash
docker build --build-arg ONNXRUNTIME_VARIANT=gpu -t faceswap-web:gpu .
```

此參數會先移除 CPU 套件，再單獨安裝同版本的 `onnxruntime-gpu`，最終環境不會同時保留兩者。InsightFace 的套件中繼資料會依賴 CPU 套件，因此非 Docker 環境請先完成 `requirements.txt` 安裝，再明確替換：

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu==1.27.0
```

不可在同一環境保留 `onnxruntime` 與 `onnxruntime-gpu`。

GPU 主機還必須提供與該 ONNX Runtime wheel 相容的 NVIDIA driver、CUDA 12.x 與 cuDNN 9.x 執行函式庫，並以 NVIDIA Container Toolkit 啟動容器。預設 Python slim image 不包含 CUDA；僅替換 Python 套件不會讓沒有 CUDA runtime 的主機取得 GPU 加速。若 CUDA provider 不可用，應用程式會退回 `CPUExecutionProvider`。

可用下列指令確認：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 模型下載與授權

- InsightFace Python library 程式碼採 MIT License；這不等於預訓練模型可自由商用。
- InsightFace 官方說明其提供的預訓練模型僅供**非商業研究用途**。`buffalo_l` 與自動／手動下載的模型都受該模型政策約束。
- `inswapper_128.onnx` 等 inswapper 系列若需商業使用、部署服務或其他授權，請先聯絡 `contact@insightface.ai` 取得書面授權。
- 下載腳本使用官方 v0.7 release、檔案大小及 SHA-256 驗證；下載中斷的檔案不會被當作可用模型。

詳見 [InsightFace 官方授權說明](https://github.com/deepinsight/insightface#license) 與 [ONNX Runtime 安裝說明](https://onnxruntime.ai/docs/install/)。部署前應自行確認實際用途符合最新模型條款。

## 常見錯誤排除

- **模型尚未準備完成**：查看容器日誌與 `GET /api/health`，確認可連線 GitHub、磁碟空間足夠，再執行 `python scripts/download_models.py --model-dir /models`。模型很大，首次建置與下載需要較長時間。
- **Zeabur build 逾時或映像過大**：重新部署以利用 build cache，並確認所選方案有足夠磁碟、記憶體與建置時間。Zeabur 不支援 Git LFS，模型應由建置腳本下載。
- **`/data` 無法寫入或資料消失**：確認 Persistent Volume 的 Mount Directory 是 `/data`，且 `DATA_DIR=/data`。掛載新 volume 會遮蔽該路徑原有內容，先備份既有資料。
- **`database is locked`**：保持 `MAX_CONCURRENT_JOBS=1`、只執行一個容器副本，並確認資料庫位於本機 Persistent Volume。
- **找不到 `CUDAExecutionProvider`**：確認只安裝 `onnxruntime-gpu`、CUDA／cuDNN 主版本相容，且容器已取得 NVIDIA GPU；否則使用預設 CPU 版。
- **沒有偵測到人臉或品質不佳**：使用清楚、光線足夠、遮擋少的正面或微側臉。`inswapper_128` 的輸入解析度有限，無法保證所有角度、遮擋與極端表情都能得到高解析結果。
- **CPU 太慢或記憶體不足**：降低 `FACE_DETECTION_SIZE`／`MAX_IMAGE_SIDE`，保持單一工作，或使用具備完整 CUDA runtime 的 GPU 主機。

## 已知限制

此專案沒有帳號、權限分流或多人隔離，適合單一受信任使用者或受控環境；若直接公開到網路，任何能開啟服務的人都可能看到或修改同一人臉庫。結果與暫存圖會定期刪除，但 `/data/faces` 與 SQLite 會持續保存，管理者應自行備份並依需求刪除。換臉模型不保證身份相似度、自然度或高解析品質，也不應用於任何身分驗證情境。

Zeabur 的 Dockerfile 與 volume 行為可參考 [Dockerfile 部署](https://zeabur.com/docs/en-US/deploy/methods/dockerfile) 與 [Volumes](https://zeabur.com/docs/en-US/data-management/volumes)。
