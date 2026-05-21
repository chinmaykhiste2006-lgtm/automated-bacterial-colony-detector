from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import numpy as np
import base64
import uvicorn
from collections import Counter

app = FastAPI()

# ===============================
# ✅ CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===============================
# 🎨 FIXED COLORS (BGR for OpenCV)
# ===============================
CLASS_COLORS_BGR = {
    "B.subtilis":    (180, 119,  31),
    "C.albicans":    (207, 190,  23),
    "Contamination": ( 44, 160,  44),
    "E.coli":        (107,  27,  13),
    "P.aeruginosa":  (194, 119, 227),
    "S.aureus":      ( 77,  77, 255)
}

# ===============================
# 🔄 LOAD YOLO
# ===============================
print("🔄 Loading YOLO model...")
yolo_model = YOLO("models/yolo/best.pt")
print("✅ YOLO Classes:", yolo_model.model.names)

# ===============================
# 🔄 LOAD UNET++ (EfficientNet-b4 + SCSE attention)
# Notebook saves state_dict only:
#   torch.save(unet.state_dict(), "best.pt")
# So we must rebuild the exact same architecture first.
# ===============================
print("🔄 Loading UNet++ model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = None

try:
    # Must match notebook Cell 5 exactly — including decoder_attention_type='scse'
    unet_model = smp.UnetPlusPlus(
        encoder_name           = "efficientnet-b4",
        encoder_weights        = None,
        in_channels            = 3,
        classes                = 1,
        activation             = None,
        decoder_attention_type = "scse"
    )

    state_dict = torch.load("models/unet/best.pt", map_location=device)
    unet_model.load_state_dict(state_dict)
    unet_model = unet_model.to(device)
    unet_model.eval()
    print(f"✅ UNet++ (EfficientNet-b4 + SCSE) loaded on: {device}")

except Exception as e:
    unet_model = None
    print(f"❌ UNet++ load failed: {e}")
    print("   ⚠️  Hybrid mode will fall back to YOLO-only")

# ===============================
# 🧠 UNET PREPROCESSING
# Matches notebook exactly:
#   - Resize to 128×128  (PATCH_SIZE = 128 in notebook)
#   - ImageNet normalization
# ===============================
unet_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# ===============================
# 🧠 RUN UNET ON A SINGLE CROP
# ===============================
def run_unet_on_crop(crop, threshold=0.5):
    """
    crop      : BGR numpy array (the YOLO-detected region)
    threshold : mask binarization threshold (0.5 matches notebook)
    returns   : binary mask (uint8, 0/255) resized back to crop dimensions
    """
    h, w = crop.shape[:2]

    # BGR → RGB (albumentations expects RGB)
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Preprocess — identical to notebook's unet_preprocess()
    transformed  = unet_transform(image=img_rgb)
    input_tensor = transformed["image"].unsqueeze(0).float().to(device)  # (1,3,128,128)

    with torch.no_grad():
        output = unet_model(input_tensor)                        # (1,1,128,128) logits
        mask   = torch.sigmoid(output).squeeze().cpu().numpy()   # (128,128) in [0,1]

    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Resize back to original crop dimensions
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return binary_mask

# ===============================
# 🔍 HYBRID: UNET REFINES EACH YOLO BOX
# ===============================
def refine_with_unet(img, detections):
    """
    For each YOLO detection:
      1. Crop the region (with small padding — PAD_FRAC=0.1)
      2. Run UNet++ on the crop
      3. Find the largest contour → fit minimum enclosing circle
      4. Replace bbox with the precise circle coordinates
    """
    if unet_model is None:
        print("⚠️  UNet not available — returning YOLO detections unchanged")
        return detections

    H, W    = img.shape[:2]
    PAD_FRAC = 0.1
    refined  = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        if x2 <= x1 or y2 <= y1:
            continue

        # Pad crop slightly
        pad_x = int((x2 - x1) * PAD_FRAC)
        pad_y = int((y2 - y1) * PAD_FRAC)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(W, x2 + pad_x)
        y2p = min(H, y2 + pad_y)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        mask = run_unet_on_crop(crop)

        # Skip if UNet found nothing meaningful
        white_ratio = np.sum(mask > 0) / mask.size
        if white_ratio < 0.05:
            continue

        # Find precise colony boundary from UNet mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            refined.append(d)   # fallback: keep original YOLO box
            continue

        # Largest contour = the colony
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        # Map circle centre back to full-image coordinates
        cx_full = int(x1p + cx)
        cy_full = int(y1p + cy)
        radius  = max(int(radius), 3)

        refined.append({
            **d,
            "circle": {
                "cx":     cx_full,
                "cy":     cy_full,
                "radius": radius
            },
            # Update bbox to tight box around the UNet circle
            "bbox": [
                max(0, cx_full - radius),
                max(0, cy_full - radius),
                min(W, cx_full + radius),
                min(H, cy_full + radius)
            ]
        })

    return refined

# ===============================
# 🎯 DRAW — boxes (YOLO) or circles (hybrid)
# ===============================
def draw_detections(img, detections, mode="yolo"):
    for d in detections:
        label = d["class"]
        conf  = d["confidence"]
        color = CLASS_COLORS_BGR.get(label, (255, 255, 255))
        text  = f"{label} {conf:.2f}"

        if mode == "hybrid" and "circle" in d:
            # ── Precise UNet circle ──
            cx = d["circle"]["cx"]
            cy = d["circle"]["cy"]
            r  = d["circle"]["radius"]

            cv2.circle(img, (cx, cy), r, color, 2)

            # Label above circle with background chip
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx = max(cx - r, 0)
            ly = max(cy - r - 5, th + 4)
            cv2.rectangle(img, (lx, ly - th - 2), (lx + tw, ly + 2), color, -1)
            cv2.putText(img, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # ── YOLO bounding box ──
            x1, y1, x2, y2 = d["bbox"]

            # Enforce minimum visible size
            if (x2 - x1) < 8:
                cx = (x1 + x2) // 2
                x1, x2 = cx - 8, cx + 8
            if (y2 - y1) < 8:
                cy = (y1 + y2) // 2
                y1, y2 = cy - 8, cy + 8

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label with background chip
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx = x1
            ly = max(y1 - 5, th + 4)
            cv2.rectangle(img, (lx, ly - th - 2), (lx + tw, ly + 2), color, -1)
            cv2.putText(img, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img

# ===============================
# 🚀 MAIN API
# ===============================
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    mode: str        = Form("yolo")   # "yolo" or "hybrid"
):
    try:
        contents = await file.read()
        if len(contents) == 0:
            return {"error": "Empty file"}

        nparr = np.frombuffer(contents, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        # ── Resize so longest side = 640 ──
        h, w  = img.shape[:2]
        scale = 640 / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)))

        # ── YOLO detection ──
        results    = yolo_model.predict(img, conf=0.25, imgsz=640, verbose=False)
        boxes      = results[0].boxes
        detections = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "id":         int(i),
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(box.conf[0]),
                "class":      str(yolo_model.model.names[int(box.cls[0])])
            })

        # ── Hybrid: refine each detection with UNet++ ──
        if mode == "hybrid":
            detections = refine_with_unet(img, detections)

        # ── Draw results ──
        img = draw_detections(img, detections, mode=mode)

        # ── Per-class counts ──
        raw_counts   = Counter([d["class"] for d in detections])
        class_counts = {k: int(v) for k, v in raw_counts.items()}

        # ── Encode to base64 JPEG ──
        _, buffer  = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_base64 = base64.b64encode(buffer).decode()

        return {
            "success":        True,
            "mode":           mode,
            "total_colonies": int(len(detections)),
            "detections":     detections,
            "class_counts":   class_counts,
            "image_base64":   f"data:image/jpeg;base64,{img_base64}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ===============================
# ▶ RUN SERVER
# ===============================
if __name__ == "__main__":
    print("🚀 Backend running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)