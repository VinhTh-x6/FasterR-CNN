import torch
import argparse
import numpy as np
import cv2
from load_model import CATEGORIES, load_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', type=str, default=None, required=True)
    parser.add_argument('--saved_checkpoint', '-o', type=str, default='trained_models/best.pt')
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3)
    args = parser.parse_args()
    return args

def predict_image(model, device, image_bgr, conf_threshold=0.3):
    """Takes a BGR image (numpy), returns the BGR image with boxes drawn + a list of results (category, score, bbox)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)) / 255.
    image = [torch.from_numpy(tensor).float().to(device)]

    with torch.no_grad():
        output = model(image)[0]

    result_image = image_bgr.copy()
    detections = []
    for bbox, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > conf_threshold:
            xmin, ymin, xmax, ymax = bbox.cpu().numpy().astype(int)
            category = CATEGORIES[label]
            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(result_image, category, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3, cv2.LINE_AA)
            detections.append({"category": category, "score": float(score), "bbox": (xmin, ymin, xmax, ymax)})

    return result_image, detections

def test_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.saved_checkpoint, device)

    ori_image = cv2.imread(args.image_path)
    result_image, _ = predict_image(model, device, ori_image, args.conf_threshold)
    cv2.imwrite("prediction.jpg", result_image)


if __name__ == '__main__':
    args = get_args()
    test_image(args)