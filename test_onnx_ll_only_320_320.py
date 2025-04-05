import os
import cv2
import argparse
import onnxruntime as ort
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

def resize_unscale(img, new_shape=(320, 320), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(ort_session, img_path, model_file, input_dim):
    inference_sv_path = os.path.splitext(img_path.replace('images', f'da_headless-det_img_dim_cmpr'))[0]
    os.makedirs(os.path.dirname(inference_sv_path), exist_ok=True)

    save_da_path = f"./{inference_sv_path}_{model_file}_da.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (input_dim, input_dim))

    img = canvas.astype(np.float32) / 255.0
    img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.transpose(2, 0, 1)[None, ...]

    # inference
    da_seg_out = ort_session.run(
        ['lane_line_seg'],
        input_feed={"images": img}
    )[0]

    # select da & ll segment area
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_da_path, da_seg_mask)
    print("detect done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop_da-640-640.onnx")
    parser.add_argument('--img_f', type=str, default="./inference/images")
    args = parser.parse_args()

    ort.set_default_logger_severity(4)

    onnx_paths = [
        './weights/yolop-320-320.onnx',
        #'./decoder_combo_experimenting/da_no-det-decoder_320-320.onnx',
       #' yolop-320-320_quantized.onnx'
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script directory
    profile_dir = os.path.join(script_dir, "profiling_results")
    os.makedirs(profile_dir, exist_ok=True)  # Ensure the directory exists

    for onnx_path in onnx_paths:
        img_dim = int(os.path.splitext(os.path.basename(onnx_path))[0].split('-')[-1])

        ort_session = ort.InferenceSession(onnx_path)
        filename = os.path.splitext(os.path.basename(onnx_path))[0]
        print(f"Load {onnx_path} done!")

        file_list = os.listdir(args.img_f)

        for i, img_name in enumerate(file_list):
            img_path = os.path.join(args.img_f, img_name)
            if os.path.isdir(img_path):
                continue

            # Profile the inference process
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.time()
            infer_yolop(ort_session, img_path=img_path, model_file=filename, input_dim=img_dim)
            end_time = time.time()

            # Save profiling results to a file
            profiler.disable()
            sanitized_img_name = img_name.replace("/", "_").replace("\\", "_")
            profile_output_file = os.path.join(profile_dir, f"profile_{sanitized_img_name}.txt")
            with open(profile_output_file, "w") as f:
                ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                ps.print_stats(10)  # Save top 10 time-consuming functions
            print(f"Profiling results saved to {profile_output_file}")

            inference_time = end_time - start_time
            print(f"Inference time for image {img_name}: {inference_time:.4f} seconds")


if __name__ == "__main__":
    main()