import cv2, glob, os
import numpy as np
import struct

def flow_to_hsv(flow):
    # Convert optical flow to HSV image for visualization
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def write_flo(filename: str, flow: np.ndarray) -> None:
    # Write optical flow to .flo file in Middlebury format
    if flow.dtype != np.float32 or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must be H×W×2 float32")
    h, w = flow.shape[:2]
    with open(filename, 'wb') as f:
        f.write(b'PIEH')
        f.write(struct.pack('<i', w))
        f.write(struct.pack('<i', h))
        f.write(flow.tobytes(order='C'))

# —— Debug: Print current working directory —— 
print("cwd:", os.getcwd())

input_dir  = "./vrig/broom-single/rgb_processed/2x/left"
output_dir = "./vrig/broom-single/flows_flo"
print("input_dir exists?", os.path.isdir(input_dir))
print("output_dir exists or created:", output_dir)
os.makedirs(output_dir, exist_ok=True)

# —— Debug: List matched image files —— 
img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
print(f"found {len(img_paths)} images:")
for p in img_paths[:5]:
    print("  ", p)
if len(img_paths) == 0:
    raise SystemExit("No .png files found — please check if input_dir is correct.")

# Create Dual TV-L1 optical flow estimator
tvl1 = cv2.optflow.createOptFlow_DualTVL1()
tvl1.setTau(0.25)
tvl1.setLambda(0.15)
tvl1.setTheta(0.3)
tvl1.setScalesNumber(5)
tvl1.setWarpingsNumber(5)
tvl1.setEpsilon(0.01)

# Compute forward and backward optical flow between consecutive frames
for i in range(len(img_paths) - 1):
    prev = cv2.imread(img_paths[i],   cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(img_paths[i+1], cv2.IMREAD_GRAYSCALE)

    flow_f = tvl1.calc(prev, curr, None).astype(np.float32)
    out_f_png = os.path.join(output_dir, f"flow_fwd_{i:04d}.png")
    out_f_flo = os.path.join(output_dir, f"flow_fwd_{i:04d}.flo")
    print(f"writing {out_f_png}, {out_f_flo}")
    cv2.imwrite(out_f_png, flow_to_hsv(flow_f))
    write_flo(out_f_flo, flow_f)

    flow_b = tvl1.calc(curr, prev, None).astype(np.float32)
    out_b_png = os.path.join(output_dir, f"flow_bwd_{i:04d}.png")
    out_b_flo = os.path.join(output_dir, f"flow_bwd_{i:04d}.flo")
    print(f"writing {out_b_png}, {out_b_flo}")
    cv2.imwrite(out_b_png, flow_to_hsv(flow_b))
    write_flo(out_b_flo, flow_b)
