import random
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_status_pil(out_bgr, text, ok=None):
    """
    Draw UTF-8 (Vietnamese) text onto an OpenCV BGR image using PIL.
    Auto scales with frame width (works well for resize_width=416).
    """
    # Color (BGR -> RGB later)
    if ok is True:
        color_rgb = (0, 200, 0)
    elif ok is False:
        color_rgb = (255, 0, 0)
    else:
        color_rgb = (255, 255, 0)

    H, W = out_bgr.shape[:2]

    # Scale font size with width (tune these if you want smaller/bigger)
    font_size = max(14, int(W * 0.045))   # ~18-20 when W=416
    pad = max(6, int(W * 0.012))
    x, y = pad, pad

    # IMPORTANT: use a font that supports Vietnamese.
    # Option A (recommended): put a font file in your repo, e.g. ./assets/DejaVuSans.ttf
    font_path = "./asset/DejaVuSans.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        # fallback if font missing (still may not support Vietnamese)
        font = ImageFont.load_default()

    # Convert OpenCV image -> PIL
    img_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Measure text and draw background box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    box_w = min(W - 2 * pad, tw + 2 * pad)
    box_h = th + 2 * pad

    # background rectangle
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))

    # If text is too long, you can wrap (simple wrap by character count)
    # For now, just draw; if it overflows, tell me and I'll add proper wrapping.
    draw.text((x + pad, y + pad), text, font=font, fill=color_rgb)

    # Convert back PIL -> OpenCV
    out_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return out_bgr
