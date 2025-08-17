from IPython.display import Markdown, display
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
from PIL import Image
import math
import torch
# import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter as _median_filter
import torch.nn.functional as F
import cv2
from torch import nn

class MaybeToTensor(nn.Module):
    def forward(self, x):
        import numpy as np
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray")

        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        return t

class AddSumChannel(nn.Module):
    """Append a 3rd channel: x[:,0] + x[:,1].
    Input:  [B, 2, H, W]
    Output: [B, 3, H, W]"""
    def forward(self, x: torch.Tensor):
        if x.dim() != 4 or x.size(1) != 2:
            raise ValueError(f"Expected [B, 2, H, W], got {tuple(x.shape)}")
        sum_ch = (x[:, 0:1] + x[:, 1:1+1])  # keep dims
        return torch.cat([x, sum_ch], dim=1)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std  = torch.as_tensor(std,  dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std",  std.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.mean.numel():
            raise ValueError(f"Channel mismatch: got C={x.size(1)}, expected {self.mean.numel()}")
        return (x - self.mean) / self.std

def normalize_flow_magnitude(flow_batch, clip_val=1e-3, eps=1e-8):
    """
    Args:
        flow_batch: torch.Tensor [B, 2, H, W]  (optical flow, u=0, v=1)
        clip_val: float, minimum threshold after normalization
    Returns:
        torch.Tensor [B, 16, 16], values in [0, 1]
    """
    # 1. Resize flow to [B, 2, 16, 16]
    flow_small = F.interpolate(flow_batch, size=(16, 16), mode="bilinear", align_corners=False)
    
    # 2. Magnitude √(u² + v²)
    u = flow_small[:, 0, :, :]
    v = flow_small[:, 1, :, :]
    mag = torch.sqrt(u ** 2 + v ** 2 + eps)  # [B, 16, 16]

    # 3. Normalize to [0,1] per sample
    mag_min = mag.view(mag.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    mag_max = mag.view(mag.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    norm_mag = (mag - mag_min) / (mag_max - mag_min + eps)

    # 4. Clip small values
    norm_mag = torch.clamp(norm_mag, min=clip_val, max=1.0)

    return norm_mag

def extract_frames(video_path, frame_indices, as_tensor=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(set([i for i in frame_indices if 0 <= i < total_frames]))
    
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # optional: convert BGR→RGB
            frames.append(frame)
        else:
            print(f"⚠️ Could not read frame {idx}")
    
    cap.release()
    
    if not frames:
        return None
    
    # Stack into numpy array [B, H, W, 3]
    batch = np.stack(frames, axis=0)
    
    if as_tensor:
        batch = torch.from_numpy(batch)  # → torch.Size([B, H, W, 3])
    
    return batch

def _segment_by_sse(X: np.ndarray, K: int):
    """
    Optimal 1D segmentation into K contiguous segments to minimize total within-segment SSE.
    X: (T, D) feature matrix (float32/64)
    Returns: boundaries list of tuples [(s0,e0), (s1,e1), ...] covering 0..T-1
    """
    T, D = X.shape
    # prefix sums for fast SSE(i,j)
    S  = np.zeros((T + 1, D), dtype=np.float64)  # sum of vectors
    S2 = np.zeros(T + 1, dtype=np.float64)       # sum of squared norms
    S[1:] = np.cumsum(X, axis=0, dtype=np.float64)
    S2[1:] = np.cumsum(np.einsum('td,td->t', X, X, optimize=True), dtype=np.float64)

    def seg_cost(i, j):
        # cost of segment [i, j] inclusive (0-based)
        L = j - i + 1
        sum_vec = S[j + 1] - S[i]
        sum_sq  = S2[j + 1] - S2[i]
        # SSE = sum||x||^2 - ||sum x||^2 / L
        return float(sum_sq - (sum_vec @ sum_vec) / L)

    # DP: dp[k][t] = min cost to segment first (t+1) items into (k+1) segments
    dp = np.full((K, T), np.inf, dtype=np.float64)
    prev = np.full((K, T), -1, dtype=np.int32)

    # base: k=0 (one segment)
    for t in range(T):
        dp[0, t] = seg_cost(0, t)
        prev[0, t] = -1

    # fill
    for k in range(1, K):
        for t in range(k, T):  # need at least k cuts -> t >= k
            best_cost = np.inf
            best_i = -1
            # try last cut at i-1, segment is [i..t]
            # small T is fine; if T is huge, consider pruning / SMAWK optimization
            for i in range(k, t + 1):
                c = dp[k - 1, i - 1] + seg_cost(i, t)
                if c < best_cost:
                    best_cost = c
                    best_i = i
            dp[k, t] = best_cost
            prev[k, t] = best_i

    # backtrack boundaries
    bounds = []
    k, t = K - 1, T - 1
    while k >= 0:
        i = prev[k, t] if k > 0 else 0
        bounds.append((i, t))
        t = i - 1
        k -= 1
    bounds.reverse()
    return bounds


def build_feature_matrix(flow, num_dir_bins=16, grid=8):
    """
    flow: np.ndarray of shape (T, 2, H, W)
    returns X: (T, D) feature matrix
    """
    T = flow.shape[0]
    feats = [flow_to_features(flow[t], num_dir_bins=num_dir_bins, grid=grid) for t in range(T)]
    return np.stack(feats, axis=0)

def cluster_flow_time_series(
    flow, n_clusters,
    num_dir_bins=16, grid=8,
    smooth_window=0, random_state=0,
    one_based=False
):
    """
    Partition time steps into n non-overlapping, contiguous clusters that cover the entire range.
    flow: np.ndarray or torch.Tensor with shape (T, 2, H, W)
    Returns:
        labels: (T,) np.int32 segment id in [0..n_clusters-1]
        clusters: list[list[int]] contiguous index lists, partition of {0..T-1} (or 1..T if one_based)
        X: (T, D) feature matrix used for segmentation
    Notes:
      - Uses optimal 1D SSE segmentation on standardized per-frame flow features.
      - Optional temporal median filtering on features for robustness: set smooth_window to an odd int (e.g., 5).
    """
    assert flow.ndim == 4 and flow.shape[1] == 2, "flow should be (T, 2, H, W)"
    T = flow.shape[0]

    # 1) Build per-frame feature matrix (you already have this)
    X = build_feature_matrix(flow, num_dir_bins=num_dir_bins, grid=grid)  # (T, D)

    # 2) Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 3) Optional temporal smoothing of features (median filter along time)
    if smooth_window and smooth_window > 1 and smooth_window % 2 == 1:
        # apply per-dimension median filter
        Xs = np.stack([_median_filter(Xs[:, d], size=smooth_window, mode='nearest') for d in range(Xs.shape[1])], axis=1)

    # 4) Optimal contiguous segmentation into n_clusters
    bounds = _segment_by_sse(Xs, n_clusters)  # list of (start, end), inclusive

    # 5) Emit labels and clusters
    labels = np.empty(T, dtype=np.int32)
    clusters = []
    for seg_id, (s, e) in enumerate(bounds):
        labels[s:e+1] = seg_id
        if one_based:
            clusters.append(list(range(s + 1, e + 2)))  # 1..T
        else:
            clusters.append(list(range(s, e + 1)))      # 0..T-1

    return labels, clusters, X


def flow_to_features(flow_t, num_dir_bins=16, grid=8):
    """
    Convert one optical-flow frame (2,H,W) into a feature vector that captures:
      - direction similarity (magnitude-weighted angle histogram)
      - spatial area of motion (pooled magnitude map)
    """
    # flow_t: (2, H, W), where [:, y, x] = (u,v)
    u, v = flow_t[0], flow_t[1]
    mag = np.sqrt(u*u + v*v)                       # (H,W)
    ang = np.arctan2(v, u)                         # [-pi, pi]
    ang = (ang + np.pi) / (2*np.pi)                # -> [0,1)
    # 1) Direction histogram (magnitude-weighted)
    bins = np.linspace(0, 1, num_dir_bins+1, endpoint=True)
    hist, _ = np.histogram(ang, bins=bins, weights=mag)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-8)

    # 2) Spatial area map (coarse pooling of magnitude)
    H, W = mag.shape
    gy, gx = grid, grid
    # Ensure divisible by grid via center crop (minimal)
    Hc, Wc = (H // gy) * gy, (W // gx) * gx
    y0, x0 = (H - Hc)//2, (W - Wc)//2
    mag_c = mag[y0:y0+Hc, x0:x0+Wc]
    # average pool to (gy,gx)
    mag_c = mag_c.reshape(gy, Hc//gy, gx, Wc//gx).mean(axis=(1,3))  # (gy,gx)
    spat = mag_c.flatten()
    if spat.sum() > 0:
        spat = spat / (np.linalg.norm(spat) + 1e-8)

    # Concatenate features
    feat = np.concatenate([hist, spat], axis=0)  # length = num_dir_bins + grid*grid
    return feat

# def build_feature_matrix(flow, num_dir_bins=16, grid=8):
#     """
#     flow: np.ndarray of shape (T, 2, H, W)
#     returns X: (T, D) feature matrix
#     """
#     T = flow.shape[0]
#     feats = [flow_to_features(flow[t], num_dir_bins=num_dir_bins, grid=grid) for t in range(T)]
#     return np.stack(feats, axis=0)

# def cluster_flow_time_series(
#     flow, n_clusters,
#     num_dir_bins=16, grid=8,
#     smooth_window=0, random_state=0
# ):
#     """
#     Cluster timesteps into n non-overlapping clusters C_i (a partition of {0..T-1}).
#     Similarity is based on per-frame direction histogram + spatial motion map.
#     - smooth_window: optional odd int (e.g., 5) to median-filter labels over time.
#     """
#     assert flow.ndim == 4 and flow.shape[1] == 2, "flow should be (T,2,H,W)"
#     X = build_feature_matrix(flow, num_dir_bins=num_dir_bins, grid=grid)

#     # Standardize features before KMeans
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
#     labels = kmeans.fit_predict(Xs)  # (T,)

#     # Optional temporal smoothing (median filter on label sequence)
#     if smooth_window and smooth_window > 1 and smooth_window % 2 == 1:
#         labels = median_filter(labels, size=smooth_window, mode='nearest')

#     # Build clusters C_i as lists of time indices
#     clusters = [np.where(labels == c)[0].tolist() for c in range(n_clusters)]
#     return labels, clusters, X  # return features too, if you want to inspect

# ---------------------------
# Example usage
    # clusters[i] is C_i, non-overlapping; sum(len(C_i)) == T

def weighted_dot(A, B):
    """
    A: [B, 16, 16, 768]
    B: [B, 16, 16]
    Returns: [B, 768]
    """
    # expand B -> [B, 16, 16, 1] for broadcasting
    B_exp = B.unsqueeze(-1)

    # elementwise multiply, then sum over spatial dims
    out = (A * B_exp).sum(dim=(1, 2))  # [B, 768]

    return out

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    # print(f"fk = {u}")
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def expand_by_edges(x, idx, out_len=100):
    """
    x: list of length B, each element is a 768-dim list/array
    idx: list of length B, unique and sorted, each in [0, out_len-1]
    Returns: list of length out_len, each element one row from x

    Policy:
      - First segment: [0 .. idx[1]] -> x[0]
      - Middle segments: (idx[k-1] .. idx[k]] -> x[k]  for k=1..B-2  (i.e., idx[k-1]+1 .. idx[k])
      - Final segment: (idx[B-2] .. end] -> x[B-1]     (i.e., idx[B-2]+1 .. out_len-1)
    """
    B = len(x)
    if len(idx) != B:
        raise ValueError("len(idx) must equal len(x)")
    if sorted(idx) != list(idx) or len(set(idx)) != B:
        raise ValueError("idx must be strictly increasing with no duplicates")
    if idx[0] < 0 or idx[-1] >= out_len:
        raise ValueError(f"All indices must lie in [0, {out_len-1}]")

    out = [None] * out_len
    if B == 1:
        return [x[0]] * out_len

    # 1) First segment: 0..idx[1]
    for j in range(0, idx[1] + 1):
        out[j] = x[0]

    # 2) Middle segments: (idx[k-1]..idx[k]] for k=1..B-2  => indices idx[k-1]+1 .. idx[k]
    for k in range(1, B - 1):
        lo = idx[k - 1] + 1
        hi = idx[k]
        for j in range(lo, hi + 1):
            out[j] = x[k]

    # 3) Final segment: (idx[B-2] .. end] => idx[B-2]+1 .. out_len-1
    for j in range(idx[B - 2] + 1, out_len):
        out[j] = x[B - 1]

    return out

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel