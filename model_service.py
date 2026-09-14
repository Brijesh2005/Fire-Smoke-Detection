"""
Model Service for Fire and Smoke Detection.
Loads the trained ResNet-50 PyTorch model and provides inference functions
for PIL Images, raw bytes, Base64 strings, OpenCV frames, and Video files.
"""

import os
import io
import time
import base64
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")


class FireSmokeDetector:
    """
    Singleton class wrapping the Fire and Smoke detection model.
    """
    _instance: Optional["FireSmokeDetector"] = None

    CLASS_NAMES: List[str] = ["Fire", "Neutral", "Smoke"]
    
    # Visual color codes (BGR for OpenCV, Hex for Web)
    COLOR_MAP = {
        "Fire": {"bgr": (0, 0, 255), "hex": "#EF4444", "hazard": True, "level": "CRITICAL"},
        "Smoke": {"bgr": (0, 165, 255), "hex": "#F59E0B", "hazard": True, "level": "WARNING"},
        "Neutral": {"bgr": (0, 255, 0), "hex": "#10B981", "hazard": False, "level": "SAFE"}
    }

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "trained-models", "model_final.pth")
        
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Image preprocessing pipeline matching training & inference
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "FireSmokeDetector":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_model(self) -> torch.nn.Module:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Loading PyTorch model from {self.model_path}...")
        try:
            # PyTorch 2.6+ defaults to weights_only=True. The model file contains a full ResNet instance.
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model = model.to(self.device)
            model.eval()

            # Ensure softmax output dimension warning is suppressed or handled
            if hasattr(model, "fc") and len(model.fc) > 3:
                # Replace unparameterized Softmax if needed to avoid warnings
                if isinstance(model.fc[3], torch.nn.Softmax):
                    model.fc[3] = torch.nn.Softmax(dim=1)

            logger.info("Model loaded and set to evaluation mode successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def predict_pil(self, img: Image.Image) -> Dict[str, Any]:
        """
        Runs inference on a PIL image and returns predictions, probabilities, and latency.
        """
        start_time = time.perf_counter()

        # Convert to RGB in case of RGBA, Grayscale, etc.
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Preprocess image
        tensor = self.transform(img)[:3, :, :].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            
            # Ensure probabilities sum to 1
            if not isinstance(output, torch.Tensor):
                output = torch.tensor(output)
            probs = torch.softmax(output, dim=1) if output.min() < 0 or output.max() > 1.05 else output
            probs_list = probs.cpu().squeeze(0).tolist()

        idx = int(np.argmax(probs_list))
        pred_class = self.CLASS_NAMES[idx]
        confidence = round(probs_list[idx] * 100.0, 2)
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

        prob_dict = {
            self.CLASS_NAMES[i]: round(probs_list[i] * 100.0, 2)
            for i in range(len(self.CLASS_NAMES))
        }

        color_info = self.COLOR_MAP.get(pred_class, self.COLOR_MAP["Neutral"])

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "is_hazard": color_info["hazard"],
            "hazard_level": color_info["level"],
            "color_hex": color_info["hex"],
            "probabilities": prob_dict,
            "latency_ms": latency_ms,
            "width": img.width,
            "height": img.height,
            "resolution": f"{img.width}x{img.height}"
        }

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.predict_pil(image)
        except Exception as e:
            logger.error(f"Error decoding image bytes: {e}")
            raise ValueError(f"Invalid image format: {e}")

    def predict_base64(self, b64_string: str) -> Dict[str, Any]:
        """
        Runs inference on a Base64-encoded image string (with or without data URI header).
        """
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_string)
            return self.predict_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

    def predict_cv2_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an OpenCV BGR frame.
        """
        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image)
        except ImportError:
            # Fallback if cv2 not available
            rgb_frame = frame[:, :, ::-1]
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image)

    def annotate_frame(self, frame: np.ndarray, pred_dict: Dict[str, Any]) -> np.ndarray:
        """
        Draws an informative status badge and text on an OpenCV frame.
        """
        try:
            import cv2
        except ImportError:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        pred_class = pred_dict.get("prediction", "Neutral")
        confidence = pred_dict.get("confidence", 0.0)
        color = self.COLOR_MAP.get(pred_class, {}).get("bgr", (0, 255, 0))

        # Top banner overlay
        overlay = annotated.copy()
        banner_height = max(50, int(h * 0.1))
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Status badge
        badge_text = f"[{pred_class.upper()}] {confidence:.1f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.6, min(1.1, w / 800))
        thickness = 2

        # Draw left indicator pill
        cv2.circle(annotated, (25, banner_height // 2), 10, color, -1)
        cv2.putText(annotated, badge_text, (45, banner_height // 2 + 7), font, font_scale, color, thickness)

        # Draw hazard border if Fire or Smoke
        if pred_dict.get("is_hazard"):
            border_thick = 4 if pred_class == "Fire" else 2
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, border_thick)

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_interval_sec: float = 0.5,
        max_duration_sec: float = 60.0
    ) -> Dict[str, Any]:
        """
        Processes a video file by sampling frames at `sample_interval_sec`,
        compiling a detection timeline and hazard statistics.
        Optionally generates an annotated video clip.
        """
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python is required for video processing.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        sample_step = max(1, int(fps * sample_interval_sec))
        max_frames_to_process = int(fps * max_duration_sec)

        timeline = []
        counts = {"Fire": 0, "Smoke": 0, "Neutral": 0}
        total_sampled = 0

        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, min(fps, 15.0), (width, height))

        frame_idx = 0
        current_pred = None

        while cap.isOpened() and frame_idx < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = round(frame_idx / fps, 2)

            # Sample periodically for ML inference
            if frame_idx % sample_step == 0 or current_pred is None:
                current_pred = self.predict_cv2_frame(frame)
                counts[current_pred["prediction"]] = counts.get(current_pred["prediction"], 0) + 1
                total_sampled += 1

                timeline.append({
                    "timestamp": timestamp_sec,
                    "frame": frame_idx,
                    "prediction": current_pred["prediction"],
                    "confidence": current_pred["confidence"],
                    "is_hazard": current_pred["is_hazard"],
                    "hazard_level": current_pred["hazard_level"]
                })

            if writer:
                annotated = self.annotate_frame(frame, current_pred)
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # Summary statistics
        fire_percent = round((counts["Fire"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0
        smoke_percent = round((counts["Smoke"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0
        neutral_percent = round((counts["Neutral"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0

        # Overall hazard assessment
        if fire_percent > 10.0:
            overall = "Fire"
        elif smoke_percent > 10.0:
            overall = "Smoke"
        else:
            overall = "Neutral"

        return {
            "status": "success",
            "duration_sec": round(duration_sec, 2),
            "processed_frames": frame_idx,
            "sampled_frames": total_sampled,
            "overall_prediction": overall,
            "is_hazard": overall in ["Fire", "Smoke"],
            "hazard_level": self.COLOR_MAP[overall]["level"],
            "stats": {
                "fire_percent": fire_percent,
                "smoke_percent": smoke_percent,
                "neutral_percent": neutral_percent,
                "counts": counts
            },
            "timeline": timeline,
            "annotated_video_saved": output_path is not None and os.path.exists(output_path)
        }
