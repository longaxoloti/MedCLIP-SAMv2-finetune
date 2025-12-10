import os, cv2, numpy as np
from glob import glob

gt_dir = r"d:\PythonCode\VQA_benchmark\MedCLIP_data\brain_tumors\test_masks"
pred_dir = r"d:\PythonCode\VQA_benchmark\MedCLIP-SAMv2\sam_outputs\brain_tumors_test_images\masks"

names = sorted(os.listdir(gt_dir))
names = [n for n in names if n.lower().endswith(('.png','.jpg','.jpeg'))]

report = []

def dice(a,b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a,b).sum()
    return (2*inter) / (a.sum() + b.sum() + 1e-8)

# sample evenly 12 files
idxs = np.linspace(0, len(names)-1, num=min(12, len(names)), dtype=int)
for i in idxs:
    n = names[i]
    gt = cv2.imread(os.path.join(gt_dir, n), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(os.path.join(pred_dir, n), cv2.IMREAD_GRAYSCALE)
    if gt is None or pr is None:
        report.append((n, 'missing'))
        continue
    pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt > 200).astype(np.uint8)
    pr_bin = (pr > 200).astype(np.uint8)
    dsc = dice(gt_bin, pr_bin)
    gt_ratio = gt_bin.mean()
    pr_ratio = pr_bin.mean()
    report.append((n, dsc, gt_ratio, pr_ratio))

for r in report:
    print(r)

# summary
if report:
    vals = [x[1] for x in report if isinstance(x[1], float)]
    if vals:
        print('Sample DSC mean:', float(np.mean(vals)))
