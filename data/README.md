# Data 目錄說明

本目錄存放 ground truth 資料。`.mat` 檔案**不進 git**，請執行下載腳本。

---

## 下載資料

```bash
python scripts/download_data.py
```

此腳本會從 [Raissi's PINN repository](https://github.com/maziarraissi/PINNs) 下載：

- `cylinder_nektar_wake.mat`（約 30 MB）

---

## 資料來源

這筆資料由 Raissi, M., Perdikaris, P., & Karniadakis, G. E. 在 [Physics-informed neural networks (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125) 中使用。原始流場由 **Nektar++**（高階譜元素法 CFD 求解器）計算，作為 PINN 的 ground truth。

---

## 資料內容

| 變數 | Shape | 說明 |
|:-----|:------|:-----|
| `U_star` | (5000, 2, 200) | 速度場；第 2 維 index 0 = u, 1 = v |
| `p_star` | (5000, 200) | 壓力場 |
| `t` | (200, 1) | 時間點（0 到 19.9，步長 0.1） |
| `X_star` | (5000, 2) | 空間座標；第 2 維 index 0 = x, 1 = y |

**空間域**：`x ∈ [1, 8]`、`y ∈ [-2, 2]`（圓柱下游矩形區）

**流場條件**：Re = 100、均勻來流速度 U∞ = 1、圓柱位於原點、直徑 D = 1

---

## 資料載入範例

```python
import scipy.io as sio
import numpy as np

data = sio.loadmat("data/cylinder_nektar_wake.mat")

U_star = data["U_star"]   # (N, 2, T)
p_star = data["p_star"]   # (N, T)
t_star = data["t"].flatten()  # (T,)
X_star = data["X_star"]   # (N, 2)

N, T = p_star.shape
print(f"{N} spatial points, {T} time steps")

# 取 t=50 的速度場
u_t50 = U_star[:, 0, 50]  # (5000,)
v_t50 = U_star[:, 1, 50]  # (5000,)
```

完整的 loader 在 `src/data/loader.py`。

---

## 資料授權

原始資料隨 Raissi et al. 2019 論文及其 GitHub 發布。本專案僅為學術與面試展示用途引用。
