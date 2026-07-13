(function () {
  "use strict";

  const API = Object.freeze({
    health: "/api/health",
    faces: "/api/faces",
    detect: "/api/detect-target-faces",
    swap: "/api/swap"
  });

  const configuredUploadMb = Number(document.body.dataset.maxUploadMb);
  const configuredImageSide = Number(document.body.dataset.maxImageSide);
  const MAX_UPLOAD_MB = Number.isFinite(configuredUploadMb) && configuredUploadMb > 0 ? configuredUploadMb : 15;
  const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
  const MAX_IMAGE_SIDE = Number.isFinite(configuredImageSide) && configuredImageSide > 0 ? configuredImageSide : 2500;
  const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
  const TERMS_STORAGE_KEY = "faceswap.termsAccepted.v1";
  const SELECTED_FACE_STORAGE_KEY = "faceswap.selectedFaceId";

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  class UserFacingError extends Error {
    constructor(message) {
      super(message);
      this.name = "UserFacingError";
    }
  }

  const state = {
    faces: [],
    selectedFaceId: safeStorageGet(SELECTED_FACE_STORAGE_KEY),
    sourceFile: null,
    sourceObjectUrl: "",
    targetFile: null,
    targetOriginalName: "",
    targetObjectUrl: "",
    targetToken: "",
    detectedFaces: [],
    targetFaceIndex: null,
    modelReady: null,
    renameFaceId: null,
    isCreating: false,
    isDetecting: false,
    isSwapping: false,
    deletingFaceIds: new Set(),
    targetRequestId: 0,
    detectController: null,
    appStarted: false
  };

  const dom = {
    mainContent: document.getElementById("mainContent"),
    libraryTitle: document.getElementById("libraryTitle"),
    serviceStatus: document.getElementById("serviceStatus"),
    serviceStatusText: document.getElementById("serviceStatusText"),
    globalAlert: document.getElementById("globalAlert"),
    globalAlertText: document.getElementById("globalAlertText"),
    dismissGlobalAlert: document.getElementById("dismissGlobalAlert"),
    createFaceForm: document.getElementById("createFaceForm"),
    faceName: document.getElementById("faceName"),
    sourceDropzone: document.getElementById("sourceDropzone"),
    sourceImageInput: document.getElementById("sourceImageInput"),
    sourcePreview: document.getElementById("sourcePreview"),
    sourcePreviewImage: document.getElementById("sourcePreviewImage"),
    sourceFileName: document.getElementById("sourceFileName"),
    clearSourceImage: document.getElementById("clearSourceImage"),
    createFaceStatus: document.getElementById("createFaceStatus"),
    saveFaceButton: document.getElementById("saveFaceButton"),
    faceCount: document.getElementById("faceCount"),
    faceListLoading: document.getElementById("faceListLoading"),
    faceEmptyState: document.getElementById("faceEmptyState"),
    faceList: document.getElementById("faceList"),
    selectedFaceImage: document.getElementById("selectedFaceImage"),
    selectedFaceFallback: document.getElementById("selectedFaceFallback"),
    selectedFaceName: document.getElementById("selectedFaceName"),
    jumpToLibrary: document.getElementById("jumpToLibrary"),
    targetDropzone: document.getElementById("targetDropzone"),
    targetImageInput: document.getElementById("targetImageInput"),
    targetPreview: document.getElementById("targetPreview"),
    targetPreviewImage: document.getElementById("targetPreviewImage"),
    targetFileInfo: document.getElementById("targetFileInfo"),
    replaceTargetImage: document.getElementById("replaceTargetImage"),
    detectionStatus: document.getElementById("detectionStatus"),
    detectedFacesSection: document.getElementById("detectedFacesSection"),
    detectedFacesHint: document.getElementById("detectedFacesHint"),
    detectedFaceCount: document.getElementById("detectedFaceCount"),
    detectedFaceList: document.getElementById("detectedFaceList"),
    restoreFace: document.getElementById("restoreFace"),
    restoreStrengthRow: document.getElementById("restoreStrengthRow"),
    restoreStrength: document.getElementById("restoreStrength"),
    restoreStrengthValue: document.getElementById("restoreStrengthValue"),
    swapButton: document.getElementById("swapButton"),
    swapRequirementHint: document.getElementById("swapRequirementHint"),
    processingPanel: document.getElementById("processingPanel"),
    resultSection: document.getElementById("resultSection"),
    resultImage: document.getElementById("resultImage"),
    processingTime: document.getElementById("processingTime"),
    downloadResult: document.getElementById("downloadResult"),
    regenerateButton: document.getElementById("regenerateButton"),
    termsDialog: document.getElementById("termsDialog"),
    termsConsent: document.getElementById("termsConsent"),
    acceptTermsButton: document.getElementById("acceptTermsButton"),
    renameDialog: document.getElementById("renameDialog"),
    renameFaceForm: document.getElementById("renameFaceForm"),
    renameFaceName: document.getElementById("renameFaceName"),
    renameStatus: document.getElementById("renameStatus"),
    confirmRenameButton: document.getElementById("confirmRenameButton"),
    cancelRenameButton: document.getElementById("cancelRenameButton"),
    dismissRenameButton: document.getElementById("dismissRenameButton"),
    toastRegion: document.getElementById("toastRegion")
  };

  function safeStorageGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (error) {
      return null;
    }
  }

  function safeStorageSet(key, value) {
    try {
      if (value === null || value === undefined || value === "") {
        window.localStorage.removeItem(key);
      } else {
        window.localStorage.setItem(key, String(value));
      }
    } catch (error) {
      // Storage may be unavailable in private browsing; the current session still works.
    }
  }

  function generateUuid() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    const randomPart = Math.random().toString(16).slice(2);
    return `${Date.now().toString(16)}-${randomPart}`;
  }

  function uploadFilename(file) {
    const extensionByType = {
      "image/jpeg": "jpg",
      "image/jpg": "jpg",
      "image/png": "png",
      "image/webp": "webp"
    };
    return `${generateUuid()}.${extensionByType[file.type] || "jpg"}`;
  }

  function openDialog(dialog, required) {
    if (!dialog) {
      return;
    }
    document.body.classList.add("modal-open");
    if (typeof dialog.showModal === "function") {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      dialog.setAttribute("open", "");
      dialog.classList.add("modal-fallback");
      if (required && dom.mainContent) {
        dom.mainContent.setAttribute("inert", "");
      }
    }
  }

  function closeDialog(dialog) {
    if (!dialog) {
      return;
    }
    if (typeof dialog.close === "function" && dialog.open) {
      dialog.close();
    } else {
      dialog.removeAttribute("open");
      dialog.classList.remove("modal-fallback");
    }
    if (dom.mainContent) {
      dom.mainContent.removeAttribute("inert");
    }
    if (!document.querySelector("dialog[open]")) {
      document.body.classList.remove("modal-open");
    }
  }

  function setButtonLoading(button, loading, loadingText) {
    const label = button.querySelector(".button-label");
    if (label && !label.dataset.defaultText) {
      label.dataset.defaultText = label.textContent;
    }
    button.classList.toggle("is-loading", loading);
    button.setAttribute("aria-busy", loading ? "true" : "false");
    if (label) {
      label.textContent = loading ? loadingText : label.dataset.defaultText;
    }
  }

  function setInlineStatus(element, message, type) {
    element.textContent = message || "";
    element.classList.remove("is-error", "is-success", "is-loading");
    if (type) {
      element.classList.add(`is-${type}`);
    }
    element.hidden = !message;
  }

  function showGlobalError(message) {
    dom.globalAlertText.textContent = message;
    dom.globalAlert.hidden = false;
  }

  function hideGlobalError() {
    dom.globalAlert.hidden = true;
    dom.globalAlertText.textContent = "";
  }

  function showToast(message, type) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.setAttribute("role", type === "error" ? "alert" : "status");
    if (type === "error") {
      toast.classList.add("is-error");
    }
    toast.textContent = message;
    dom.toastRegion.appendChild(toast);
    window.setTimeout(function () {
      toast.remove();
    }, 4200);
  }

  function extractErrorMessage(payload) {
    if (!payload) {
      return "";
    }
    if (typeof payload === "string") {
      return payload;
    }
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail.map(function (item) {
        return item && item.msg ? item.msg : "";
      }).filter(Boolean).join("；");
    }
    if (typeof payload.error === "string") {
      return payload.error;
    }
    if (typeof payload.message === "string") {
      return payload.message;
    }
    return "";
  }

  function localizeError(error, fallback) {
    if (error instanceof UserFacingError) {
      return error.message;
    }

    if (error && error.name === "AbortError") {
      return "操作已取消，請重新選擇照片後再試一次。";
    }

    const status = error instanceof ApiError ? error.status : 0;
    const detail = error instanceof ApiError ? extractErrorMessage(error.payload) || error.message : (error && error.message ? error.message : "");
    const normalized = detail.toLowerCase();
    const errorCode = error instanceof ApiError && error.payload && typeof error.payload.code === "string" ? error.payload.code.toLowerCase() : "";

    if (status === 413 || normalized.includes("too large") || normalized.includes("file size") || normalized.includes("15mb")) {
      return `圖片超過 ${MAX_UPLOAD_MB} MB，請壓縮或改用較小的照片。`;
    }
    if (status === 415 || normalized.includes("unsupported") || normalized.includes("file type") || normalized.includes("format")) {
      return "圖片格式不支援，請使用 JPG、PNG 或 WEBP。";
    }
    if ((status === 503 && errorCode.includes("model")) || (normalized.includes("model") && (normalized.includes("ready") || normalized.includes("load") || normalized.includes("download")))) {
      return "模型尚未準備完成，請稍後再試；若持續發生，請檢查模型下載狀態。";
    }
    if (normalized.includes("no face") || normalized.includes("face not detected") || normalized.includes("cannot detect")) {
      return "照片中沒有偵測到清楚的人臉，請改用正面、光線充足的照片。";
    }
    if (normalized.includes("multiple") && normalized.includes("face")) {
      return "偵測到多張臉，請先選擇要替換的人臉。";
    }
    if (normalized.includes("token") || normalized.includes("expired") || normalized.includes("target not found")) {
      return "目標照片已過期或不存在，請重新上傳照片。";
    }
    if ((normalized.includes("source") && normalized.includes("not found")) || errorCode === "face_record_not_found") {
      return "選擇的來源臉已不存在，請重新選擇。";
    }
    if (normalized.includes("database") && normalized.includes("locked")) {
      return "人臉庫目前忙碌中，請稍後再試。";
    }
    if (status === 404) {
      return "找不到指定的資料，可能已被清除，請重新操作。";
    }
    if (status === 429) {
      return "目前處理中的工作較多，請稍候再試。";
    }
    if (error instanceof TypeError) {
      return "無法連線到服務，請檢查網路後再試。";
    }
    if (/[㐀-鿿]/.test(detail)) {
      return detail;
    }
    if (status >= 500) {
      return fallback || "伺服器處理失敗，請稍後再試。";
    }
    return fallback || "操作失敗，請稍後再試。";
  }

  async function apiRequest(path, options) {
    const requestOptions = options || {};
    const timeoutMs = requestOptions.timeoutMs || 60000;
    const controller = new AbortController();
    let timedOut = false;
    let externalAbortHandler = null;

    if (requestOptions.signal) {
      if (requestOptions.signal.aborted) {
        controller.abort();
      } else {
        externalAbortHandler = function () {
          controller.abort();
        };
        requestOptions.signal.addEventListener("abort", externalAbortHandler, { once: true });
      }
    }

    const timer = window.setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    try {
      const headers = new Headers(requestOptions.headers || {});
      headers.set("X-Requested-With", "faceswap-web");
      const response = await window.fetch(path, {
        method: requestOptions.method || "GET",
        headers: headers,
        body: requestOptions.body,
        signal: controller.signal,
        credentials: "same-origin"
      });

      const rawText = response.status === 204 ? "" : await response.text();
      let payload = null;
      if (rawText) {
        try {
          payload = JSON.parse(rawText);
        } catch (error) {
          payload = rawText;
        }
      }

      if (!response.ok) {
        throw new ApiError(extractErrorMessage(payload) || `HTTP ${response.status}`, response.status, payload);
      }
      return payload;
    } catch (error) {
      if (timedOut) {
        throw new UserFacingError("伺服器回應逾時，照片可能較大或目前工作較多，請稍後再試。");
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
      if (requestOptions.signal && externalAbortHandler) {
        requestOptions.signal.removeEventListener("abort", externalAbortHandler);
      }
    }
  }

  function safeImageUrl(value, allowBase64) {
    if (typeof value !== "string" || !value.trim()) {
      return "";
    }
    const candidate = value.trim();
    if (allowBase64 && /^data:image\/(?:jpeg|jpg|png|webp);base64,[a-z0-9+/=\s]+$/i.test(candidate)) {
      return candidate;
    }
    if (allowBase64 && !candidate.includes(":") && !candidate.includes("/") && /^[a-z0-9+/=\s]+$/i.test(candidate) && candidate.length > 40) {
      return `data:image/jpeg;base64,${candidate}`;
    }
    try {
      const url = new URL(candidate, window.location.origin);
      if (url.protocol !== "http:" && url.protocol !== "https:") {
        return "";
      }
      return url.href;
    } catch (error) {
      return "";
    }
  }

  function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
      return "";
    }
    if (bytes < 1024 * 1024) {
      return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatDate(value) {
    if (!value) {
      return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    try {
      return new Intl.DateTimeFormat("zh-TW", {
        year: "numeric",
        month: "short",
        day: "numeric"
      }).format(date);
    } catch (error) {
      return date.toLocaleDateString();
    }
  }

  function firstCharacter(value) {
    const characters = Array.from(String(value || ""));
    return characters.length ? characters[0].toUpperCase() : "臉";
  }

  function normalizeFace(rawFace) {
    if (!rawFace || rawFace.id === undefined || rawFace.id === null) {
      return null;
    }
    return {
      id: String(rawFace.id),
      name: String(rawFace.name || "未命名人臉"),
      thumbnailUrl: safeImageUrl(rawFace.thumbnail_url || rawFace.thumbnail || "", true),
      createdAt: rawFace.created_at || ""
    };
  }

  function makeImageContainer(className, imageUrl, altText, fallbackText, lazy) {
    const container = document.createElement("div");
    container.className = className;

    const fallback = document.createElement("span");
    fallback.className = "image-fallback";
    fallback.textContent = firstCharacter(fallbackText);

    if (imageUrl) {
      const image = document.createElement("img");
      image.src = imageUrl;
      image.alt = altText;
      image.decoding = "async";
      if (lazy) {
        image.loading = "lazy";
      }
      fallback.hidden = true;
      image.addEventListener("error", function () {
        image.hidden = true;
        fallback.hidden = false;
      }, { once: true });
      container.appendChild(image);
    }
    container.appendChild(fallback);
    return container;
  }

  function setServiceStatus(kind, message) {
    dom.serviceStatus.classList.remove("is-checking", "is-ready", "is-warning", "is-error");
    dom.serviceStatus.classList.add(`is-${kind}`);
    dom.serviceStatusText.textContent = message;
  }

  async function loadHealth() {
    try {
      const data = await apiRequest(API.health, { timeoutMs: 15000 });
      const explicitReady = data && (data.model_ready !== undefined ? data.model_ready : (data.ready !== undefined ? data.ready : (data.model && data.model.ready)));
      const status = String(data && data.status ? data.status : "").toLowerCase();
      const provider = data && (data.provider || data.execution_provider);

      if (explicitReady === false || status === "degraded" || status === "error") {
        state.modelReady = false;
        setServiceStatus("warning", "模型尚未就緒");
        const modelError = data && typeof data.model_error === "string" ? data.model_error.trim() : "";
        showGlobalError(modelError ? `模型尚未準備完成：${modelError}` : "模型尚未準備完成，請稍後重新整理；若持續發生，請檢查模型下載狀態。" );
      } else if (explicitReady === true || ["ok", "healthy", "ready"].includes(status)) {
        state.modelReady = true;
        setServiceStatus("ready", "服務已就緒");
      } else {
        state.modelReady = null;
        setServiceStatus("ready", "服務已連線");
      }

      if (provider) {
        dom.serviceStatus.title = `運算提供者：${provider}`;
      }
    } catch (error) {
      state.modelReady = null;
      setServiceStatus("error", "服務連線異常");
    } finally {
      updateCreateButton();
      updateSwapButton();
    }
  }

  async function loadFaces() {
    dom.faceListLoading.hidden = false;
    dom.faceEmptyState.hidden = true;
    dom.faceList.hidden = true;

    try {
      const data = await apiRequest(API.faces, { timeoutMs: 20000 });
      const rawFaces = Array.isArray(data) ? data : (data && Array.isArray(data.faces) ? data.faces : []);
      state.faces = rawFaces.map(normalizeFace).filter(Boolean);

      if (state.selectedFaceId && !state.faces.some(function (face) { return face.id === String(state.selectedFaceId); })) {
        state.selectedFaceId = null;
      }
      if (!state.selectedFaceId && state.faces.length === 1) {
        state.selectedFaceId = state.faces[0].id;
      }
      safeStorageSet(SELECTED_FACE_STORAGE_KEY, state.selectedFaceId);
      renderFaces();
    } catch (error) {
      state.faces = [];
      dom.faceListLoading.hidden = true;
      dom.faceList.hidden = true;
      dom.faceEmptyState.hidden = false;
      const title = dom.faceEmptyState.querySelector("strong");
      const message = dom.faceEmptyState.querySelector("p");
      title.textContent = "無法載入人臉庫";
      message.textContent = "請確認服務狀態後重新整理頁面。";
      showGlobalError(localizeError(error, "無法載入人臉庫，請稍後重新整理頁面。"));
      updateSelectedFaceDisplay();
    }
  }

  function renderFaces() {
    dom.faceListLoading.hidden = true;
    dom.faceCount.textContent = `${state.faces.length} 張`;
    dom.faceList.replaceChildren();

    const emptyTitle = dom.faceEmptyState.querySelector("strong");
    const emptyMessage = dom.faceEmptyState.querySelector("p");
    emptyTitle.textContent = "人臉庫還是空的";
    emptyMessage.textContent = "先在上方新增一張清楚的人臉照片。";

    if (!state.faces.length) {
      dom.faceEmptyState.hidden = false;
      dom.faceList.hidden = true;
      updateSelectedFaceDisplay();
      return;
    }

    const fragment = document.createDocumentFragment();
    state.faces.forEach(function (face) {
      const selected = face.id === String(state.selectedFaceId);
      const deleting = state.deletingFaceIds.has(face.id);
      const actionsDisabled = deleting || state.isSwapping;
      const card = document.createElement("article");
      card.className = "face-card";
      card.setAttribute("role", "listitem");
      if (selected) {
        card.classList.add("is-selected");
      }

      card.appendChild(makeImageContainer("face-thumbnail", face.thumbnailUrl, `${face.name}的人臉縮圖`, face.name, true));

      const body = document.createElement("div");
      body.className = "face-card-body";

      const titleRow = document.createElement("div");
      titleRow.className = "face-card-title-row";
      const name = document.createElement("h4");
      name.className = "face-card-name";
      name.textContent = face.name;
      name.title = face.name;
      titleRow.appendChild(name);
      if (selected) {
        const badge = document.createElement("span");
        badge.className = "selected-badge";
        badge.textContent = "已選擇";
        titleRow.appendChild(badge);
      }
      body.appendChild(titleRow);

      const formattedDate = formatDate(face.createdAt);
      if (formattedDate) {
        const meta = document.createElement("span");
        meta.className = "face-card-meta";
        meta.textContent = `建立於 ${formattedDate}`;
        body.appendChild(meta);
      }

      const actions = document.createElement("div");
      actions.className = "face-card-actions";

      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "face-action";
      selectButton.textContent = selected ? "目前使用" : "選擇這張臉";
      selectButton.setAttribute("aria-pressed", selected ? "true" : "false");
      selectButton.disabled = actionsDisabled;
      selectButton.addEventListener("click", function () {
        selectFace(face.id, true);
      });

      const renameButton = document.createElement("button");
      renameButton.type = "button";
      renameButton.className = "face-action";
      renameButton.textContent = "重新命名";
      renameButton.disabled = actionsDisabled;
      renameButton.addEventListener("click", function () {
        openRenameDialog(face);
      });

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "face-action danger";
      deleteButton.textContent = deleting ? "刪除中…" : "刪除";
      deleteButton.disabled = actionsDisabled;
      deleteButton.addEventListener("click", function () {
        deleteFace(face);
      });

      actions.append(selectButton, renameButton, deleteButton);
      body.appendChild(actions);
      card.appendChild(body);
      fragment.appendChild(card);
    });

    dom.faceList.appendChild(fragment);
    dom.faceEmptyState.hidden = true;
    dom.faceList.hidden = false;
    updateSelectedFaceDisplay();
  }

  function selectFace(faceId, notify) {
    const face = state.faces.find(function (item) {
      return item.id === String(faceId);
    });
    if (!face) {
      showToast("找不到這張人臉，請重新整理人臉庫。", "error");
      return;
    }
    state.selectedFaceId = face.id;
    safeStorageSet(SELECTED_FACE_STORAGE_KEY, face.id);
    renderFaces();
    updateSwapButton();
    if (notify) {
      showToast(`已選擇「${face.name}」`);
    }
  }

  function updateSelectedFaceDisplay() {
    const selectedFace = state.faces.find(function (face) {
      return face.id === String(state.selectedFaceId);
    });

    dom.selectedFaceImage.onerror = null;
    if (!selectedFace) {
      dom.selectedFaceName.textContent = "尚未選擇";
      dom.selectedFaceImage.hidden = true;
      dom.selectedFaceImage.removeAttribute("src");
      dom.selectedFaceImage.alt = "";
      dom.selectedFaceFallback.hidden = false;
      dom.selectedFaceFallback.textContent = "?";
      dom.jumpToLibrary.textContent = "前往選擇";
      updateSwapButton();
      return;
    }

    dom.selectedFaceName.textContent = selectedFace.name;
    dom.selectedFaceFallback.textContent = firstCharacter(selectedFace.name);
    dom.jumpToLibrary.textContent = "更換";

    if (selectedFace.thumbnailUrl) {
      dom.selectedFaceImage.src = selectedFace.thumbnailUrl;
      dom.selectedFaceImage.alt = `${selectedFace.name}的人臉縮圖`;
      dom.selectedFaceImage.hidden = false;
      dom.selectedFaceFallback.hidden = true;
      dom.selectedFaceImage.onerror = function () {
        dom.selectedFaceImage.hidden = true;
        dom.selectedFaceFallback.hidden = false;
      };
    } else {
      dom.selectedFaceImage.hidden = true;
      dom.selectedFaceImage.removeAttribute("src");
      dom.selectedFaceFallback.hidden = false;
    }
    updateSwapButton();
  }

  async function createFace(event) {
    event.preventDefault();
    const name = dom.faceName.value.trim();
    if (!name) {
      setInlineStatus(dom.createFaceStatus, "請輸入人臉名稱。", "error");
      dom.faceName.focus();
      return;
    }
    if (!state.sourceFile) {
      setInlineStatus(dom.createFaceStatus, "請先選擇一張來源臉照片。", "error");
      dom.sourceImageInput.focus();
      return;
    }
    if (state.modelReady === false) {
      setInlineStatus(dom.createFaceStatus, "模型尚未準備完成，請稍後再試。", "error");
      return;
    }

    state.isCreating = true;
    updateCreateButton();
    setButtonLoading(dom.saveFaceButton, true, "正在建立人臉…");
    setInlineStatus(dom.createFaceStatus, "正在偵測並保存人臉，請稍候…", "loading");

    const formData = new FormData();
    formData.append("name", name);
    formData.append("image", state.sourceFile, uploadFilename(state.sourceFile));

    try {
      const data = await apiRequest(API.faces, {
        method: "POST",
        body: formData,
        timeoutMs: 180000
      });
      const createdFace = normalizeFace(data && data.face ? data.face : data);
      clearSourceImage();
      dom.faceName.value = "";
      setInlineStatus(dom.createFaceStatus, `「${name}」已儲存到人臉庫。`, "success");
      if (createdFace) {
        state.selectedFaceId = createdFace.id;
        safeStorageSet(SELECTED_FACE_STORAGE_KEY, createdFace.id);
      }
      await loadFaces();
      showToast(`已建立「${name}」`);
    } catch (error) {
      setInlineStatus(dom.createFaceStatus, localizeError(error, "建立人臉失敗，請確認照片中有清楚的單一人臉。"), "error");
    } finally {
      state.isCreating = false;
      setButtonLoading(dom.saveFaceButton, false, "");
      updateCreateButton();
    }
  }

  function openRenameDialog(face) {
    state.renameFaceId = face.id;
    dom.renameFaceName.value = face.name;
    setInlineStatus(dom.renameStatus, "", "");
    openDialog(dom.renameDialog, false);
    window.setTimeout(function () {
      dom.renameFaceName.focus();
      dom.renameFaceName.select();
    }, 0);
  }

  async function renameFace(event) {
    event.preventDefault();
    const face = state.faces.find(function (item) {
      return item.id === String(state.renameFaceId);
    });
    const newName = dom.renameFaceName.value.trim();
    if (!face) {
      setInlineStatus(dom.renameStatus, "這張人臉已不存在，請重新整理。", "error");
      return;
    }
    if (!newName) {
      setInlineStatus(dom.renameStatus, "請輸入新名稱。", "error");
      dom.renameFaceName.focus();
      return;
    }
    if (newName === face.name) {
      closeDialog(dom.renameDialog);
      return;
    }

    dom.confirmRenameButton.disabled = true;
    setButtonLoading(dom.confirmRenameButton, true, "儲存中…");
    setInlineStatus(dom.renameStatus, "", "");

    try {
      const data = await apiRequest(`${API.faces}/${encodeURIComponent(face.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
        timeoutMs: 30000
      });
      const updated = normalizeFace(data && data.face ? data.face : data);
      face.name = updated ? updated.name : newName;
      if (updated && updated.thumbnailUrl) {
        face.thumbnailUrl = updated.thumbnailUrl;
      }
      closeDialog(dom.renameDialog);
      renderFaces();
      showToast(`已重新命名為「${face.name}」`);
    } catch (error) {
      setInlineStatus(dom.renameStatus, localizeError(error, "重新命名失敗，請稍後再試。"), "error");
    } finally {
      dom.confirmRenameButton.disabled = false;
      setButtonLoading(dom.confirmRenameButton, false, "");
    }
  }

  async function deleteFace(face) {
    const confirmed = window.confirm(`確定要刪除「${face.name}」嗎？此動作無法復原。`);
    if (!confirmed) {
      return;
    }
    state.deletingFaceIds.add(face.id);
    renderFaces();
    try {
      await apiRequest(`${API.faces}/${encodeURIComponent(face.id)}`, {
        method: "DELETE",
        timeoutMs: 30000
      });
      if (state.selectedFaceId === face.id) {
        state.selectedFaceId = null;
        safeStorageSet(SELECTED_FACE_STORAGE_KEY, null);
      }
      state.faces = state.faces.filter(function (item) {
        return item.id !== face.id;
      });
      if (!state.selectedFaceId && state.faces.length === 1) {
        state.selectedFaceId = state.faces[0].id;
        safeStorageSet(SELECTED_FACE_STORAGE_KEY, state.selectedFaceId);
      }
      showToast(`已刪除「${face.name}」`);
    } catch (error) {
      showToast(localizeError(error, "刪除失敗，請稍後再試。"), "error");
    } finally {
      state.deletingFaceIds.delete(face.id);
      renderFaces();
    }
  }

  function updateCreateButton() {
    const valid = Boolean(dom.faceName.value.trim() && state.sourceFile && !state.isCreating && !state.isSwapping && state.modelReady !== false);
    dom.saveFaceButton.disabled = !valid;
  }

  function validateImageFile(file) {
    if (!file) {
      throw new UserFacingError("請選擇一張圖片。" );
    }
    if (!ALLOWED_IMAGE_TYPES.has(String(file.type || "").toLowerCase())) {
      throw new UserFacingError("圖片格式不支援，請使用 JPG、PNG 或 WEBP。" );
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      throw new UserFacingError(`圖片超過 ${MAX_UPLOAD_MB} MB，請壓縮或改用較小的照片。`);
    }
    if (file.size <= 0) {
      throw new UserFacingError("圖片內容是空的，請重新選擇檔案。" );
    }
  }

  async function decodeImageSource(file) {
    if (typeof window.createImageBitmap === "function") {
      try {
        const bitmap = await window.createImageBitmap(file, { imageOrientation: "from-image" });
        return {
          source: bitmap,
          width: bitmap.width,
          height: bitmap.height,
          cleanup: function () { bitmap.close(); }
        };
      } catch (error) {
        // Fall through for browsers that do not support imageOrientation options.
      }
    }

    const objectUrl = URL.createObjectURL(file);
    try {
      const image = await new Promise(function (resolve, reject) {
        const element = new Image();
        element.onload = function () { resolve(element); };
        element.onerror = function () { reject(new UserFacingError("無法讀取圖片內容，檔案可能已損壞或格式不符。")); };
        element.src = objectUrl;
      });
      return {
        source: image,
        width: image.naturalWidth,
        height: image.naturalHeight,
        cleanup: function () { URL.revokeObjectURL(objectUrl); }
      };
    } catch (error) {
      URL.revokeObjectURL(objectUrl);
      throw error;
    }
  }

  function canvasToBlob(canvas, type) {
    return new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) {
          resolve(blob);
        } else {
          reject(new UserFacingError("無法縮小圖片，請改用較小的照片。"));
        }
      }, type, type === "image/png" ? undefined : 0.95);
    });
  }

  async function prepareImageFile(file) {
    validateImageFile(file);
    const decoded = await decodeImageSource(file);
    try {
      if (!decoded.width || !decoded.height) {
        throw new UserFacingError("無法讀取圖片尺寸，請重新選擇照片。" );
      }
      if (Math.max(decoded.width, decoded.height) <= MAX_IMAGE_SIDE) {
        return {
          file: file,
          width: decoded.width,
          height: decoded.height,
          resized: false
        };
      }

      const scale = MAX_IMAGE_SIDE / Math.max(decoded.width, decoded.height);
      const width = Math.max(1, Math.round(decoded.width * scale));
      const height = Math.max(1, Math.round(decoded.height * scale));
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { alpha: file.type === "image/png" });
      if (!context) {
        throw new UserFacingError("瀏覽器無法處理這張圖片，請改用較小的照片。" );
      }
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(decoded.source, 0, 0, width, height);
      const outputType = file.type === "image/png" ? "image/png" : (file.type === "image/webp" ? "image/webp" : "image/jpeg");
      const resizedBlob = await canvasToBlob(canvas, outputType);
      canvas.width = 1;
      canvas.height = 1;
      return {
        file: resizedBlob,
        width: width,
        height: height,
        resized: true
      };
    } finally {
      decoded.cleanup();
    }
  }

  async function handleSourceFile(file) {
    setInlineStatus(dom.createFaceStatus, "正在讀取照片…", "loading");
    try {
      const prepared = await prepareImageFile(file);
      if (state.sourceObjectUrl) {
        URL.revokeObjectURL(state.sourceObjectUrl);
      }
      state.sourceFile = prepared.file;
      state.sourceObjectUrl = URL.createObjectURL(prepared.file);
      dom.sourcePreviewImage.src = state.sourceObjectUrl;
      dom.sourceFileName.textContent = `${file.name} · ${formatFileSize(prepared.file.size)}`;
      dom.sourcePreview.hidden = false;
      dom.sourceDropzone.hidden = true;
      if (prepared.resized) {
        setInlineStatus(dom.createFaceStatus, `照片已等比例縮小至 ${prepared.width} × ${prepared.height} 像素。`, "success");
      } else {
        setInlineStatus(dom.createFaceStatus, "", "");
      }
    } catch (error) {
      clearSourceImage();
      setInlineStatus(dom.createFaceStatus, localizeError(error, "無法讀取來源臉照片。"), "error");
    } finally {
      updateCreateButton();
    }
  }

  function clearSourceImage() {
    if (state.sourceObjectUrl) {
      URL.revokeObjectURL(state.sourceObjectUrl);
    }
    state.sourceFile = null;
    state.sourceObjectUrl = "";
    dom.sourcePreviewImage.removeAttribute("src");
    dom.sourcePreview.hidden = true;
    dom.sourceDropzone.hidden = false;
    dom.sourceImageInput.value = "";
    updateCreateButton();
  }

  function clearResult() {
    dom.resultSection.hidden = true;
    dom.resultImage.removeAttribute("src");
    dom.downloadResult.href = "#";
    dom.processingTime.textContent = "";
  }

  function resetDetection() {
    state.targetToken = "";
    state.detectedFaces = [];
    state.targetFaceIndex = null;
    dom.detectedFaceList.replaceChildren();
    dom.detectedFacesSection.hidden = true;
    setInlineStatus(dom.detectionStatus, "", "");
    updateSwapButton();
  }

  function clearTargetImage() {
    state.targetRequestId += 1;
    if (state.detectController) {
      state.detectController.abort();
      state.detectController = null;
    }
    if (state.targetObjectUrl) {
      URL.revokeObjectURL(state.targetObjectUrl);
    }
    state.targetFile = null;
    state.targetOriginalName = "";
    state.targetObjectUrl = "";
    state.isDetecting = false;
    dom.targetPreviewImage.removeAttribute("src");
    dom.targetPreview.hidden = true;
    dom.targetDropzone.hidden = false;
    dom.targetImageInput.value = "";
    resetDetection();
    clearResult();
  }

  async function handleTargetFile(file) {
    const requestId = state.targetRequestId + 1;
    clearTargetImage();
    state.targetRequestId = requestId;
    state.isDetecting = true;
    updateSwapButton();
    setInlineStatus(dom.detectionStatus, "正在讀取照片…", "loading");

    try {
      const prepared = await prepareImageFile(file);
      if (requestId !== state.targetRequestId) {
        return;
      }
      state.targetFile = prepared.file;
      state.targetOriginalName = file.name;
      state.targetObjectUrl = URL.createObjectURL(prepared.file);
      dom.targetPreviewImage.src = state.targetObjectUrl;
      dom.targetPreview.hidden = false;
      dom.targetDropzone.hidden = true;
      const resizeNote = prepared.resized ? ` · 已縮小至 ${prepared.width} × ${prepared.height}` : "";
      dom.targetFileInfo.textContent = `${file.name} · ${formatFileSize(prepared.file.size)}${resizeNote}`;
      await detectTargetFaces(requestId);
    } catch (error) {
      if (requestId !== state.targetRequestId || (error && error.name === "AbortError")) {
        return;
      }
      setInlineStatus(dom.detectionStatus, localizeError(error, "無法讀取或偵測目標照片。"), "error");
    } finally {
      if (requestId === state.targetRequestId) {
        state.isDetecting = false;
        updateSwapButton();
      }
    }
  }

  function normalizeDetectedFace(rawFace, fallbackIndex) {
    const item = rawFace || {};
    const parsedIndex = Number(item.index !== undefined ? item.index : fallbackIndex);
    return {
      index: Number.isFinite(parsedIndex) ? parsedIndex : fallbackIndex,
      thumbnailUrl: safeImageUrl(item.thumbnail_url || item.thumbnail || item.crop_url || item.crop || "", true),
      bbox: Array.isArray(item.bbox) ? item.bbox : (Array.isArray(item.position) ? item.position : null)
    };
  }

  async function detectTargetFaces(requestId) {
    if (!state.targetFile) {
      return;
    }
    if (state.detectController) {
      state.detectController.abort();
    }
    state.detectController = new AbortController();
    resetDetection();
    setInlineStatus(dom.detectionStatus, "正在偵測目標照片中的人臉…", "loading");

    const formData = new FormData();
    formData.append("image", state.targetFile, uploadFilename(state.targetFile));

    const data = await apiRequest(API.detect, {
      method: "POST",
      body: formData,
      signal: state.detectController.signal,
      timeoutMs: 180000
    });

    if (requestId !== state.targetRequestId) {
      return;
    }

    const token = data && (data.target_token || data.token);
    const rawFaces = data && (Array.isArray(data.faces) ? data.faces : (Array.isArray(data.detected_faces) ? data.detected_faces : []));
    const reportedCount = Number(data && (data.face_count !== undefined ? data.face_count : data.count));
    const faceCount = Number.isFinite(reportedCount) ? reportedCount : rawFaces.length;

    if (!token) {
      throw new UserFacingError("伺服器沒有回傳目標照片識別碼，請重新上傳照片。" );
    }
    if (faceCount < 1) {
      throw new UserFacingError("照片中沒有偵測到清楚的人臉，請改用正面、光線充足的照片。" );
    }

    const usableRawFaces = rawFaces.length ? rawFaces : Array.from({ length: faceCount }, function (_, index) {
      return { index: index };
    });
    state.targetToken = String(token);
    state.detectedFaces = usableRawFaces.map(normalizeDetectedFace);
    state.targetFaceIndex = faceCount === 1 ? state.detectedFaces[0].index : null;
    renderDetectedFaces();

    if (faceCount === 1) {
      setInlineStatus(dom.detectionStatus, "已偵測到 1 張臉，並自動選取。", "success");
    } else {
      setInlineStatus(dom.detectionStatus, `偵測到 ${faceCount} 張臉，請選擇要替換的對象。`, "");
    }
  }

  function renderDetectedFaces() {
    dom.detectedFaceList.replaceChildren();
    const faceCount = state.detectedFaces.length;
    dom.detectedFaceCount.textContent = `${faceCount} 張`;
    dom.detectedFacesHint.textContent = faceCount === 1 ? "已自動選取偵測到的人臉。" : "偵測到多張臉，請選擇其中一張。";

    const fragment = document.createDocumentFragment();
    state.detectedFaces.forEach(function (face, position) {
      const selected = face.index === state.targetFaceIndex;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "detected-face-button";
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", selected ? "true" : "false");
      button.setAttribute("aria-label", `目標臉 ${position + 1}${selected ? "，已選擇" : ""}`);
      button.tabIndex = selected || (state.targetFaceIndex === null && position === 0) ? 0 : -1;

      button.appendChild(makeImageContainer("detected-thumbnail", face.thumbnailUrl, `目標臉 ${position + 1} 的縮圖`, String(position + 1), false));

      const label = document.createElement("span");
      label.className = "detected-face-label";
      const labelText = document.createElement("span");
      labelText.textContent = `臉 ${position + 1}`;
      const radioDot = document.createElement("span");
      radioDot.className = "radio-dot";
      radioDot.setAttribute("aria-hidden", "true");
      label.append(labelText, radioDot);
      button.appendChild(label);

      button.addEventListener("click", function () {
        selectTargetFace(face.index);
      });
      button.addEventListener("keydown", function (event) {
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
          return;
        }
        event.preventDefault();
        const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
        const nextPosition = (position + direction + state.detectedFaces.length) % state.detectedFaces.length;
        selectTargetFace(state.detectedFaces[nextPosition].index, true);
      });
      fragment.appendChild(button);
    });

    dom.detectedFaceList.appendChild(fragment);
    dom.detectedFacesSection.hidden = false;
    updateSwapButton();
  }

  function selectTargetFace(faceIndex, focusSelected) {
    state.targetFaceIndex = faceIndex;
    renderDetectedFaces();
    if (focusSelected) {
      const selectedButton = dom.detectedFaceList.querySelector('[aria-checked="true"]');
      if (selectedButton) {
        selectedButton.focus();
      }
    }
  }

  function updateSwapButton() {
    const hasSource = Boolean(state.selectedFaceId && state.faces.some(function (face) {
      return face.id === String(state.selectedFaceId);
    }));
    const hasTarget = Boolean(state.targetToken);
    const hasTargetFace = state.targetFaceIndex !== null && state.targetFaceIndex !== undefined;
    const ready = hasSource && hasTarget && hasTargetFace && !state.isDetecting && !state.isSwapping && state.modelReady !== false;
    dom.swapButton.disabled = !ready;

    if (state.modelReady === false) {
      dom.swapRequirementHint.textContent = "模型尚未準備完成，請稍後再試。";
    } else if (!hasSource) {
      dom.swapRequirementHint.textContent = "請先從人臉庫選擇一張來源臉。";
    } else if (!state.targetFile) {
      dom.swapRequirementHint.textContent = "請上傳一張含有人臉的目標照片。";
    } else if (state.isDetecting) {
      dom.swapRequirementHint.textContent = "正在偵測目標照片中的人臉…";
    } else if (!hasTarget) {
      dom.swapRequirementHint.textContent = "目標照片尚未完成偵測，請重新上傳。";
    } else if (!hasTargetFace) {
      dom.swapRequirementHint.textContent = "偵測到多張臉，請選擇要替換的對象。";
    } else if (state.isSwapping) {
      dom.swapRequirementHint.textContent = "正在處理照片，請稍候。";
    } else {
      dom.swapRequirementHint.textContent = "設定完成，可以開始換臉。";
    }
  }

  function formatProcessingTime(data) {
    const milliseconds = Number(data && data.processing_time_ms);
    if (Number.isFinite(milliseconds) && milliseconds >= 0) {
      return `處理時間 ${milliseconds < 1000 ? Math.round(milliseconds) + " 毫秒" : (milliseconds / 1000).toFixed(1) + " 秒"}`;
    }
    const seconds = Number(data && data.processing_time);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return `處理時間 ${seconds.toFixed(seconds < 10 ? 1 : 0)} 秒`;
    }
    if (data && typeof data.processing_time === "string" && data.processing_time.trim()) {
      return `處理時間 ${data.processing_time.trim()}`;
    }
    return "";
  }

  async function performSwap(regenerating) {
    updateSwapButton();
    if (!state.selectedFaceId || !state.targetToken || state.targetFaceIndex === null || state.targetFaceIndex === undefined) {
      showGlobalError("請先選擇來源臉、上傳目標照片，並選定要替換的人臉。" );
      return;
    }

    state.isSwapping = true;
    hideGlobalError();
    clearResult();
    dom.processingPanel.hidden = false;
    dom.targetImageInput.disabled = true;
    dom.replaceTargetImage.disabled = true;
    dom.restoreFace.disabled = true;
    dom.restoreStrength.disabled = true;
    setButtonLoading(dom.swapButton, true, regenerating ? "正在重新生成…" : "正在換臉…");
    setButtonLoading(dom.regenerateButton, regenerating, "正在重新生成…");
    dom.regenerateButton.disabled = true;
    renderFaces();
    updateCreateButton();
    updateSwapButton();

    const formData = new FormData();
    formData.append("face_id", String(state.selectedFaceId));
    formData.append("target_token", state.targetToken);
    formData.append("target_face_index", String(state.targetFaceIndex));
    formData.append("restore_face", dom.restoreFace.checked ? "true" : "false");
    formData.append("restore_strength", dom.restoreStrength.value);

    try {
      const data = await apiRequest(API.swap, {
        method: "POST",
        body: formData,
        timeoutMs: 600000
      });
      const resultUrl = safeImageUrl(data && (data.result_url || data.url), false);
      const downloadUrl = safeImageUrl(data && (data.download_url || data.result_url || data.url), false);
      if (!resultUrl || !downloadUrl) {
        throw new UserFacingError("伺服器沒有回傳有效的結果圖片，請重新生成。" );
      }

      dom.resultImage.src = resultUrl;
      dom.resultImage.onerror = function () {
        showGlobalError("結果圖片已過期或載入失敗，請重新生成。" );
      };
      dom.downloadResult.href = downloadUrl;
      dom.downloadResult.setAttribute("download", "換臉結果.jpg");
      dom.processingTime.textContent = formatProcessingTime(data);
      dom.resultSection.hidden = false;
      showToast(regenerating ? "已重新生成結果" : "換臉完成");
      window.setTimeout(function () {
        dom.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    } catch (error) {
      const message = localizeError(error, "換臉失敗，請確認照片與來源臉後再試一次。" );
      showGlobalError(message);
      showToast("換臉未完成", "error");
    } finally {
      state.isSwapping = false;
      dom.processingPanel.hidden = true;
      dom.targetImageInput.disabled = false;
      dom.replaceTargetImage.disabled = false;
      dom.restoreFace.disabled = false;
      dom.restoreStrength.disabled = false;
      setButtonLoading(dom.swapButton, false, "");
      setButtonLoading(dom.regenerateButton, false, "");
      dom.regenerateButton.disabled = false;
      renderFaces();
      updateCreateButton();
      updateSwapButton();
    }
  }

  function bindDropzone(dropzone, input, handler) {
    ["dragenter", "dragover"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (eventName === "dragleave" && event.relatedTarget && dropzone.contains(event.relatedTarget)) {
          return;
        }
        dropzone.classList.remove("is-dragging");
      });
    });
    dropzone.addEventListener("drop", function (event) {
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) {
        return;
      }
      handler(files[0]);
    });
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) {
        handler(input.files[0]);
      }
    });
  }

  function bindEvents() {
    dom.dismissGlobalAlert.addEventListener("click", hideGlobalError);
    dom.faceName.addEventListener("input", updateCreateButton);
    dom.createFaceForm.addEventListener("submit", createFace);
    dom.clearSourceImage.addEventListener("click", function () {
      clearSourceImage();
      setInlineStatus(dom.createFaceStatus, "", "");
    });
    bindDropzone(dom.sourceDropzone, dom.sourceImageInput, handleSourceFile);
    bindDropzone(dom.targetDropzone, dom.targetImageInput, handleTargetFile);

    dom.jumpToLibrary.addEventListener("click", function () {
      dom.libraryTitle.scrollIntoView({ behavior: "smooth", block: "start" });
      window.setTimeout(function () {
        if (state.faces.length) {
          const selectButton = dom.faceList.querySelector(".face-action");
          if (selectButton) {
            selectButton.focus();
          }
        } else {
          dom.faceName.focus();
        }
      }, 350);
    });

    dom.replaceTargetImage.addEventListener("click", function () {
      dom.targetImageInput.value = "";
      dom.targetImageInput.click();
    });
    dom.restoreFace.addEventListener("change", function () {
      dom.restoreStrengthRow.hidden = !dom.restoreFace.checked;
    });
    dom.restoreStrength.addEventListener("input", function () {
      dom.restoreStrengthValue.textContent = `${Math.round(Number(dom.restoreStrength.value) * 100)}%`;
    });
    dom.swapButton.addEventListener("click", function () {
      performSwap(false);
    });
    dom.regenerateButton.addEventListener("click", function () {
      performSwap(true);
    });

    dom.renameFaceForm.addEventListener("submit", renameFace);
    [dom.cancelRenameButton, dom.dismissRenameButton].forEach(function (button) {
      button.addEventListener("click", function () {
        closeDialog(dom.renameDialog);
      });
    });
    dom.renameDialog.addEventListener("close", function () {
      state.renameFaceId = null;
      document.body.classList.remove("modal-open");
    });

    dom.termsConsent.addEventListener("change", function () {
      dom.acceptTermsButton.disabled = !dom.termsConsent.checked;
    });
    dom.acceptTermsButton.addEventListener("click", function () {
      if (!dom.termsConsent.checked) {
        return;
      }
      safeStorageSet(TERMS_STORAGE_KEY, "accepted");
      closeDialog(dom.termsDialog);
      startApp();
    });
    dom.termsDialog.addEventListener("cancel", function (event) {
      event.preventDefault();
    });

    window.addEventListener("beforeunload", function () {
      if (state.sourceObjectUrl) {
        URL.revokeObjectURL(state.sourceObjectUrl);
      }
      if (state.targetObjectUrl) {
        URL.revokeObjectURL(state.targetObjectUrl);
      }
    });
  }

  function startApp() {
    if (state.appStarted) {
      return;
    }
    state.appStarted = true;
    Promise.allSettled([loadHealth(), loadFaces()]);
  }

  function initialize() {
    bindEvents();
    updateCreateButton();
    updateSwapButton();
    if (safeStorageGet(TERMS_STORAGE_KEY) === "accepted") {
      startApp();
    } else {
      dom.termsConsent.checked = false;
      dom.acceptTermsButton.disabled = true;
      openDialog(dom.termsDialog, true);
    }
  }

  initialize();
})();
