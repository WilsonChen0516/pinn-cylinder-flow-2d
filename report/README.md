# 技術報告

本目錄存放專案的技術報告。最終產出是 `report.pdf`。

---

## 報告大綱

| 章節 | 頁數 | 內容 |
|:-----|:----:|:-----|
| 1. Motivation | 0.5 | 為何研究 PINN × CFD、本專案定位 |
| 2. Problem Formulation | 1 | 2D NS 方程、無因次化、三個實驗設定 |
| 3. PINN Methodology | 1.5 | 網路架構、loss 設計、兩階段訓練、Stream function |
| 4. Results | 3 | E1/E2/E3 結果，含關鍵圖 |
| 5. Analysis | 1 | 空間誤差分布、收斂行為 |
| 6. Discussion | 1 | 遇到的問題、PINN 的限制、vs 傳統 CFD |
| 7. Path to Deployment | 0.5 | 延伸到 Modulus / Omniverse 的規劃 |
| 8. Conclusion | 0.5 | 總結 |
| Appendix | - | Hyperparameters、訓練時間、reproducibility 資訊 |

**目標總頁數**：6-8 頁（A4，單欄，11pt）

---

## 撰寫原則

### 1. 誠實大於技巧

- 寫清楚實驗的 limitation（例如 E1 的 BC 是 Dirichlet 不是 no-slip）
- 失敗結果也要報，特別是 MLP baseline 在 N=100 時徹底失敗——這證明 PINN 的價值
- 不要說「達到 state-of-the-art」，這題是 2019 paper、不需要吹牛

### 2. 每張圖要獨立看得懂

- Caption 至少 3 行，說明圖顯示什麼、怎麼看、結論是什麼
- 圖內文字、axis label、colorbar 都要清楚

### 3. 為 hiring manager 優化

- **最重要的結論在 Abstract 第一句**
- **最好看的圖放 Motivation 或 Results 的開頭**
- 最後一節寫「如何延伸到 Modulus/Omniverse 場景」——這是針對職位的 targeted content

---

## LaTeX 模板建議

```bash
# 使用 Overleaf 或本地 TeXLive
# 建議模板：article 或 IEEE conference（簡潔）
```

推薦套件：

```latex
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{subcaption}       % 多子圖
\usepackage{booktabs}         % 漂亮的表格
\usepackage{hyperref}         % 超連結
\usepackage[utf8]{inputenc}   % 中文支援
\usepackage{xeCJK}            % 若用 XeLaTeX + 中文
```

---

## 產出流程

```bash
# 本地編譯
cd report/
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```

最終 `report.pdf` commit 進 git（面試官 clone 後直接能看，不需裝 LaTeX）。

---

## 圖片管理

報告的所有圖來自 `../figures/`，用相對路徑引用：

```latex
\includegraphics[width=0.8\textwidth]{../figures/fig_e3_data_efficiency.png}
```

動畫（mp4/gif）無法嵌入 PDF，改在 report 中放**代表性 frame**，並在 caption 註明：

```
Full animation available at: <GitHub repo URL>/figures/anim_*.mp4
```

---

## 最後一節的模板

這節是本報告相對於一般 PINN tutorial 的差異化，也是針對面試職位的 targeted content。

```
7. Path to Production Deployment

While this work demonstrates PINN feasibility on a benchmark problem,
production deployment in digital-twin contexts (e.g., NVIDIA Modulus
on Omniverse) requires additional engineering:

7.1 Scaling to 3D and higher Re
  - Domain decomposition (XPINN, FBPINN)
  - Fourier feature networks for spectral bias
  - ...

7.2 Integration with simulation platforms
  - Modulus's built-in Symbolic equation system
  - Omniverse USD for geometry import
  - ...

7.3 Real-time inference
  - Model distillation to smaller networks
  - ONNX export for deployment
  - ...
```

**這節的目的**是告訴面試官：「我知道這個 demo 只是起點，我有規劃好要往哪裡走」。
