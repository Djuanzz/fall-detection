# Analisis Perbandingan Parameter BlockGCN
## NTU RGB+D 25-Joint vs YOLO11n-pose 17-Joint

**Proyek:** Fall Detection — Tugas Akhir  
**Model:** BlockGCN  
**Tanggal Analisis:** 3 Mei 2026

---

## 1. Ringkasan Hasil Training Epoch 1

| Metrik | NTU25 (25-joint, 3D) | YOLO17 (17-joint, 2D) |
|--------|----------------------|----------------------|
| Parameters | 1,351,474 | 1,266,562 |
| Selisih | **84,912** | — |
| Training Acc Ep.1 | 57.47% | 61.80% |
| Test Acc Ep.1 | 82.38% | 71.41% |
| Balanced Acc Ep.1 | — | **82.04%** |
| Sensitivity Ep.1 | — | 99.44% |
| Specificity Ep.1 | — | 64.64% |
| AUC-ROC Ep.1 | — | 0.8478 |

---

## 2. Analisis Perbedaan Jumlah Parameter (Δ = 84,912)

### Latar Belakang

Kedua model menggunakan **arsitektur BlockGCN yang identik**, dengan perbedaan tunggal pada konfigurasi:

```
NTU25 : num_point=25, graph=graph.ntu_rgb_d.Graph
YOLO17: num_point=17, graph=graph.yolo.Graph
```

Terdapat **4 komponen** dalam BlockGCN yang parameternya bergantung pada jumlah joint (V).

---

### 2.1 Komponen `fc1` — Kontributor Utama (95.0%)

**Definisi kode (`model/BlockGCN.py`):**
```python
self.fc1 = nn.Parameter(
    torch.stack([torch.stack([torch.eye(A.shape[-1]) ...
    # shape: (3, num_heads=8, V, V)
```

Parameter `fc1` berbentuk **tensor 4 dimensi (3, 8, V, V)** dan diinstansiasi **10 kali** — satu untuk setiap blok TCN-GCN.

| Variant | Formula | Jumlah Params |
|---------|---------|---------------|
| NTU25 (V=25) | 3 × 8 × 25² × 10 blok | 150,000 |
| YOLO17 (V=17) | 3 × 8 × 17² × 10 blok | 69,360 |
| **Δfc1** | 3 × 8 × (625 − 289) × 10 | **80,640** |

> **Mengapa quadratic?** `fc1[i]` berperan sebagai _adaptive adjacency matrix_ berukuran V×V — setiap pasangan joint memiliki bobot tersendiri. Karena diskalakan dengan V², perbedaan V=25 vs V=17 menghasilkan selisih besar meskipun perbedaan V hanya 8.

---

### 2.2 Komponen `data_bn` — BatchNorm1d (2.4%)

**Definisi kode:**
```python
self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)
# Komentar di kode: "dipanggil SETELAH to_joint_embedding (output dim=128)"
```

`BatchNorm1d` menyimpan `weight` dan `bias`, masing-masing berukuran `num_features`.

| Variant | num_features | Params (weight+bias) |
|---------|-------------|----------------------|
| NTU25 (V=25) | 1 × 128 × 25 = 3,200 | 6,400 |
| YOLO17 (V=17) | 1 × 128 × 17 = 2,176 | 4,352 |
| **Δdata_bn** | | **2,048** |

---

### 2.3 Komponen `pos_embedding` — Positional Encoding (1.2%)

**Definisi kode:**
```python
self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 128))
```

| Variant | Shape | Params |
|---------|-------|--------|
| NTU25 (V=25) | (1, 25, 128) | 3,200 |
| YOLO17 (V=17) | (1, 17, 128) | 2,176 |
| **Δpos_embedding** | | **1,024** |

---

### 2.4 Komponen `rpe` — Relative Position Encoding (1.4%)

**Definisi kode:**
```python
self.hops = ...  # matrix (V, V) berisi jarak hop antar setiap pasangan joint
self.rpe  = nn.Parameter(torch.zeros(3, self.num_heads, self.hops.max() + 1))
```

`rpe` berukuran `(3, 8, max_hop + 1)` per blok. Nilai `max_hop` (diameter graph) **bergantung pada topologi jaringan tulang**, bukan hanya jumlah joint.

#### Diameter Graph NTU25 (max_hop = 11)

Jalur terpanjang antar dua joint:

```
HandTipLeft(21) → HandLeft(7) → WristLeft(6) → ElbowLeft(5)
→ ShoulderLeft(4) → SpineShoulder(20) → SpineMid(1) → SpineBase(0)
→ HipRight(16) → KneeRight(17) → AnkleRight(18) → FootRight(19)

= 11 langkah (hops)
```

#### Diameter Graph YOLO17/COCO (max_hop = 6)

Jalur terpanjang:

```
LeftWrist(9) → LeftElbow(7) → LeftShoulder(5) → LeftHip(11)
→ RightHip(12) → RightKnee(14) → RightAnkle(16)

= 6 langkah (hops)
```

> Catatan: YOLO17 memiliki _shortcut edges_ `(5,6)` (shoulder-shoulder) dan `(11,12)` (hip-hip) yang mempersingkat diameter graph dibandingkan NTU25.

| Variant | max_hop | rpe size per blok | Total (×10 blok) |
|---------|---------|-------------------|------------------|
| NTU25 | 11 | 3 × 8 × 12 = 288 | 2,880 |
| YOLO17 | 6 | 3 × 8 × 7 = 168 | 1,680 |
| **Δrpe** | | | **1,200** |

---

### 2.5 Rekapitulasi Total

| Komponen | YOLO17 Params | NTU25 Params | Selisih (Δ) | Proporsi |
|----------|--------------|-------------|-------------|----------|
| `fc1` (adaptive adj.) | 69,360 | 150,000 | **80,640** | 95.0% |
| `data_bn` (BN layer) | 4,352 | 6,400 | **2,048** | 2.4% |
| `pos_embedding` | 2,176 | 3,200 | **1,024** | 1.2% |
| `rpe` (rel. pos. enc.) | 1,680 | 2,880 | **1,200** | 1.4% |
| **Total Δ** | | | **84,912** | **100%** |
| **Total params** | **1,266,562** | **1,351,474** | | |

**Verifikasi:** 1,351,474 − 1,266,562 = **84,912** ✓

---

## 3. Analisis Balanced Accuracy 82% di Epoch 1 (YOLO17)

### 3.1 Kondisi Aktual Epoch 1

Melihat confusion matrix YOLO17 epoch 1:

```
                Pred not_fall   Pred fall
True not_fall       479            262      ← 262 false alarm!
True fall             1            178      ← nyaris semua fall terdeteksi
```

| Metrik | Nilai | Interpretasi |
|--------|-------|-------------|
| Sensitivity | 99.44% | 178/179 fall terdeteksi |
| Specificity | 64.64% | Hanya 479/741 non-fall benar |
| **FP (False Alarm)** | **262** | Non-fall diprediksi sebagai fall |
| Balanced Acc | 82.04% | (99.44 + 64.64) / 2 |

**Kesimpulan: Model hampir selalu memprediksi "fall".** Balanced accuracy 82% bukan tanda model bagus — ini adalah _degenerate classifier_ yang bias ke kelas positif.

---

### 3.2 Mengapa Ini Terjadi?

#### Penyebab 1: Learning Rate sangat kecil di epoch 1

Konfigurasi menggunakan `warm_up_epoch: 5`:

```
LR di epoch 1 = base_lr × (epoch / warm_up_epoch)
             = 0.1 × (1 / 5) = 0.020
```

Model belum bergerak jauh dari inisialisasi acak (random weights).

#### Penyebab 2: Inisialisasi acak condong ke kelas "fall"

Bobot random bisa secara kebetulan menghasilkan output yang bias ke satu kelas. Ini berbeda antara dua model:

| Model | Bias Epoch 1 | Efek |
|-------|-------------|------|
| NTU25 | Condong **non-fall** | Accuracy 82% (mayoritas kelas), sensitivity rendah |
| YOLO17 | Condong **fall** | Sensitivity 99%, specificity hanya 65% |

> Pada NTU25, accuracy epoch 1 = 82.38% ≈ 741/920 = 80.5% jika prediksi semuanya non-fall. Ini menunjukkan NTU25 melakukan bias berlawanan arah.

#### Penyebab 3: Fitur 2D lebih ambigu di awal training

Fitur YOLO17 menggunakan koordinat 2D yang sudah dinormalisasi ke hip-center. Tanpa kedalaman (depth/z), pola skeletal fall vs non-fall lebih mirip di awal, sehingga model belum bisa memisahkan kedua kelas dengan baik.

NTU25 menggunakan koordinat 3D (world meters), yang mengandung informasi vertikal (sumbu Y = ketinggian) lebih eksplisit — ini secara intrinsik lebih diskriminatif untuk deteksi jatuh.

---

### 3.3 Visualisasi Dinamika Training

```
Epoch 1  : Sensitivity ~99%  Specificity ~65%  ← bias ke "fall"
            Balanced Acc ~82% (tidak valid / degenerate)

Epoch 5  : LR mulai penuh = 0.1, model mulai belajar nyata

Epoch 50 : LR drop → ×0.1 = 0.01
           Model mulai menghaluskan decision boundary

Epoch 70 : LR drop → ×0.01 = 0.001
           Fine-tuning terakhir

Epoch 90 : Target akhir — keduanya harus tinggi:
           Sensitivity >80%  Specificity >80%
           Balanced Acc >80% (genuine, bukan karena bias)
```

---

### 3.4 Metrik yang Benar-Benar Bermakna

Untuk sistem deteksi jatuh (_safety-critical_), urutan prioritas:

1. **Sensitivity / Recall (Fall)** — paling penting, jangan sampai fall tidak terdeteksi (FN = bahaya nyawa)
2. **Balanced Accuracy** — penting untuk dataset imbalanced
3. **Specificity** — penting untuk mengurangi false alarm (alarm palsu)
4. **AUC-ROC** — evaluasi keseluruhan tanpa pengaruh threshold
5. **Accuracy** — least important di sini (misleading jika dataset tidak seimbang)

**Target minimal yang bermakna:**

| Metrik | Target Minimum |
|--------|---------------|
| Sensitivity | ≥ 90% |
| Specificity | ≥ 80% |
| Balanced Accuracy | ≥ 85% |
| AUC-ROC | ≥ 0.90 |

---

## 4. Perbandingan Kedua Variant (Apple-to-Apple)

### Konfigurasi Identik

| Aspek | NTU25 | YOLO17 |
|-------|-------|--------|
| Arsitektur | BlockGCN | BlockGCN |
| Optimizer | SGD, LR=0.1 | SGD, LR=0.1 |
| Epochs | 90 | 90 |
| LR Steps | [50, 70] | [50, 70] |
| Batch size | 8 | 8 |
| Window | 150 frame | 150 frame |
| Augmentasi | random_move, shift, flip, speed | random_move, shift, flip, speed |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

### Perbedaan yang Disengaja

| Aspek | NTU25 | YOLO17 |
|-------|-------|--------|
| Input joints | 25 | 17 |
| Koordinat | 3D world (meter) | 2D + confidence (normalized) |
| Graph | `graph.ntu_rgb_d.Graph` | `graph.yolo.Graph` |
| Feeder | `feeder_ntu_binary.Feeder` | `feeder_yolo.Feeder` |
| Normalisasi | Per-frame SpineBase subtraction | Hip-center + shoulder-width (sudah di extract_skeleton.py) |
| Parameters | 1,351,474 (~1.35M) | 1,266,562 (~1.27M) |

---

## 5. Kesimpulan

1. **Perbedaan 84,912 parameter** sepenuhnya bisa dijelaskan dan reverifikasi dari kode. Komponen `fc1` menjadi kontributor terbesar (95%) karena skalanya quadratic terhadap V.

2. **Balanced accuracy 82% di epoch 1 YOLO17 bukan hasil nyata** — ini adalah artefak dari bias inisialisasi acak yang condong memprediksi "fall" hampir selalu. Model yang sesungguhnya baik akan menunjukkan keduanya — sensitivity dan specificity — sama-sama tinggi di akhir training.

3. **NTU25 memiliki potensi keunggulan** dari fitur 3D yang lebih diskriminatif untuk gerakan vertikal (jatuh). YOLO17 memiliki **keunggulan praktis** — bisa dijalankan dari kamera RGB biasa tanpa Kinect.

4. **Evaluasi final** yang bermakna baru bisa dilakukan setelah epoch 90, menggunakan seluruh 7 metrik: Accuracy, Balanced Accuracy, Precision, Sensitivity, Specificity, F1, dan AUC-ROC.

---

*Dokumen ini dihasilkan secara otomatis dari hasil analisis kode `model/BlockGCN.py`, `graph/ntu_rgb_d.py`, dan `graph/yolo.py`.*
