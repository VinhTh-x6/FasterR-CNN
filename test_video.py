import torch
import argparse
import numpy as np
import cv2
import os
import subprocess
import uuid
import tempfile
from load_model import CATEGORIES, load_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", "-v", required=True)
    parser.add_argument("--saved_checkpoint", '-o', default="trained_models/best.pt")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3)
    parser.add_argument("--show", action="store_true", help="Show a preview window (CLI use only)")
    return parser.parse_args()

def _get_ffmpeg_exe():
    """
    Get the path to the ffmpeg binary. Prefer imageio-ffmpeg (bundled with
    the pip package, runs immediately on Windows without needing a
    separate ffmpeg install or PATH edit). If imageio-ffmpeg isn't
    installed, fall back to the "ffmpeg" command on the system PATH
    (requires the user to install it themselves).
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _reencode_to_h264(src_path, dst_path):
    """
    Re-encode the video file to H.264 (yuv420p) using ffmpeg so browsers
    can play it. cv2.VideoWriter with fourcc 'avc1'/'h264' on most
    pip-installed opencv-python builds does NOT have a real H.264 encoder
    (licensing constraints) — it writes a broken .mp4 file without
    raising an error, and the browser shows a player that won't play
    anything (stuck at 0:00, black frame).
    """
    ffmpeg_exe = _get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", src_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",       # required — many browsers can't read yuv444/422
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",   # allows playback to start before the whole file downloads
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Run: pip install imageio-ffmpeg "
            "(or install ffmpeg manually and add it to PATH)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg re-encode failed: {e.stderr.decode(errors='ignore')}")

def process_video(model, device, video_path, output_path, conf_threshold=0.3,
                   progress_callback=None, show=False):
    """Read the video from video_path, draw boxes on each frame, and write to output_path (H.264, web-playable)."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_size = 1280
    scale = max_size / max(height, width)
    if scale < 1:
        out_width, out_height = int(width * scale), int(height * scale)
    else:
        out_width, out_height = width, height

    # Write to a TEMPORARY file using mp4v — will be re-encoded to H.264 in the next step.
    tmp_raw_path = os.path.join(tempfile.gettempdir(), f"raw_{uuid.uuid4().hex}.mp4")
    writer = cv2.VideoWriter(tmp_raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to initialize VideoWriter for temp file '{tmp_raw_path}'. "
            "Check whether the mp4v codec is available in the installed OpenCV build."
        )

    frame_idx = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            max_size = 1280
            scale = max_size / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, (2, 0, 1)) / 255.0
            image = [torch.from_numpy(rgb).float().to(device)]

            output = model(image)[0]
            boxes, labels, scores = output["boxes"], output["labels"], output["scores"]

            for box, label, score in zip(boxes, labels, scores):
                if score < conf_threshold:
                    continue
                xmin, ymin, xmax, ymax = box.cpu().numpy().astype(int)
                category = CATEGORIES[label.item()]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(frame, f"{category}: {score:.2f}", (xmin, max(20, ymin - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            writer.write(frame)

            if show:
                cv2.imshow("Prediction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # Re-encode the temp file to real H.264, then return output_path
    _reencode_to_h264(tmp_raw_path, output_path)
    if os.path.exists(tmp_raw_path):
        os.remove(tmp_raw_path)

def test_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.saved_checkpoint, device)
    process_video(model, device, args.video_path, "result.mp4", args.conf_threshold, show=args.show)


if __name__ == "__main__":
    args = get_args()
    test_video(args)