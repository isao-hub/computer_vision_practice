# 髭検出・修復アプリケーション v6

髭を検出し、MAT/LaMa Inpainting で自然に除去・色補正するGradioアプリケーション。

## 新機能

- **MAT (Mask-Aware Transformer)**: CVPR 2022 Best Paper Finalist
- **顔画像に特化**: FFHQとCelebA-HQの2つのプリトレーニングモデルに対応（512x512処理）
- **髭オーバーレイ**: 色補正後に残す髭を元画像からオーバーレイ（自然な薄め効果）

## 機能

- **ルールベース検出**: 髭を1本ずつ高精度に検出（推奨）
- **Grounded SAM**: テキストプロンプトによる自動髭検出（オプション）
- **矩形選択 / 自由形状**: 白ブラシで検出領域を指定
- **座標入力**: 数値で正確に領域を指定可能
- **カラーハイライト**: 検出した各髭を異なる色でハイライト表示
- **削除対象選択**: ランダム/面積大/面積小/信頼度順で削除対象を選択（赤色で強調）
- **複数Inpainting手法**: MAT (FFHQ/CelebA-HQ), LaMa, OpenCV
- **色調補正**: LAB色空間ベースの青髭補正 + 髭オーバーレイ

## 必要な環境

- Python 3.10 以上（3.11 または 3.12 推奨）
- CUDA 対応 GPU（推奨、CPUでも動作可能）

## インストール

### 1. 必須パッケージのインストール

```bash
pip install gradio numpy opencv-python pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install simple-lama-inpainting
```

### 2. MAT Inpainting（顔画像専用・高品質）

MAT を使用する場合のみ必要です。

**チェックポイントのダウンロード:**

[Hugging Face - MAT-inpainting-fp16](https://huggingface.co/spacepxl/MAT-inpainting-fp16/blob/main/README.md) から以下のファイルをダウンロードしてください。

| ファイル | 説明 | 配置場所 |
|---------|------|---------|
| `MAT_FFHQ_512_fp16.safetensors` | 一般的な顔画像向け | `image-inpainting/checkpoints/mat/` |
| `MAT_CelebA-HQ_512_fp16.safetensors` | セレブ顔データセット向け | `image-inpainting/checkpoints/mat/` |

### 3. オプション: Grounded SAM（高度な検出機能）

Grounded SAM を使用する場合のみ必要です。ルールベース検出のみ使用する場合は不要です。

```bash
pip install segment-anything groundingdino-py
```

**チェックポイントのダウンロード:**

| ファイル | ダウンロードURL | 配置場所 |
|---------|----------------|---------|
| `sam_vit_h_4b8939.pth` | [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `image-inpainting/` |
| `groundingdino_swint_ogc.pth` | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO/releases) | `image-inpainting/` または親フォルダ |

## ファイル構成

### 必須ファイル

```
image-inpainting/
├── app_mat_coloring.py       # メインアプリケーション（v6）
├── utils.py                  # BeardRemovalPipelineクラス
├── config.py                 # 設定ファイル（MAX_IMAGE_SIZE, FEATHER_RADIUS等）
├── app.py                    # LaMa単体のImage inpaintingアプリ
├── app_grounded_sam_lama.py  # LamaとSAMによる髭除去シミュレーションアプリ
├── core/                     # コアモジュール
│   ├── __init__.py
│   ├── inpainting.py         # LaMa Inpainting ラッパー（SimpleLama使用）
│   └── image_utils.py        # 画像処理ユーティリティ
└── checkpoints/
│   └── mat/
│       ├── MAT_FFHQ_512_fp16.safetensors      # FFHQ モデル
│       └── MAT_CelebA-HQ_512_fp16.safetensors # CelebA-HQ モデル
├── requirements.txt          # 依存パッケージ一覧
└── README.md                 # このファイル
```

### オプション（Grounded SAM 使用時）

```
image-inpainting/
├── sam_vit_h_4b8939.pth      # SAM チェックポイント（約2.4GB）
└── groundingdino_swint_ogc.pth  # Grounding DINO チェックポイント（約694MB）
    ※ 親フォルダに配置しても自動検出されます
```

## 使い方

### LaMa単体の動作検証 (app.py)

```bash
python app.py
```

### メインアプリ起動（v6）

```bash
cd image-inpainting
python app_mat_coloring.py
```

ブラウザで `http://127.0.0.1:7867` にアクセス。

### 起動時のログメッセージ

起動時に以下のようなログが表示されます：

```
PyTorch: 利用可能 (CUDA: True/False)
SAM: 利用可能
Grounding DINO: 利用可能
LaMa Inpainting: 利用可能
```

**各項目の意味:**

| メッセージ | 説明 |
|-----------|------|
| `PyTorch: 利用可能 (CUDA: True)` | GPU で高速処理が可能 |
| `PyTorch: 利用可能 (CUDA: False)` | CPU モードで動作（やや遅いが問題なし） |
| `SAM: 利用可能` | Grounded SAM 検出モードが使用可能 |
| `Grounding DINO: 利用可能` | Grounded SAM 検出モードが使用可能 |
| `LaMa Inpainting: 利用可能` | 髭除去機能が使用可能（必須） |

### 無視してよい警告メッセージ

以下の警告は動作に影響しないため、無視して構いません：

```
FutureWarning: Importing from timm.models.layers is deprecated...
```
→ `timm` パッケージの内部警告。将来のバージョンで解消予定。

```
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```
→ Grounding DINO のカスタム CUDA 演算子が未コンパイル。CPU で代替処理されるため問題なし。

```
DeprecationWarning: The 'theme' parameter...
```
→ Gradio の API 変更に関する警告。動作に影響なし。

### ワークフロー

#### Tab 1: 髭検出

1. **画像をアップロード**
2. **髭の範囲を選択**
   - 矩形: 白色ブラシで塗りつぶし
   - 自由形状: 線で囲む
   - 座標入力: 数値で正確に指定
3. **検出モードを選択**
   - ルールベース（1本ずつ検出）: 推奨、高精度
   - Grounded SAM: テキストプロンプトベース
4. **「髭を検出」をクリック**
   - 各髭が異なる色でハイライト表示される
5. **Remove % で削除対象を選択**
   - スライダーで削除割合を指定
   - 選択モード: ランダム / 面積大 / 面積小 / 信頼度順
   - 削除対象は赤色で強調表示
   - **残りの髭はTab 3でオーバーレイ可能**
6. **「マスクを Tab 2 に転送」をクリック**

#### Tab 2: 髭除去（Inpainting）

1. **インペインティング手法を選択**
   - MAT (FFHQ - 顔専用): 一般的な顔画像向け（推奨）
   - MAT (CelebA-HQ - 顔専用): セレブ顔データセット向け
   - Simple LaMa: 汎用的なInpainting
   - OpenCV Telea / Navier-Stokes: 軽量な補間
2. **MAT強化モード**（オプション）
   - テクスチャ強度: 元画像の肌質感を復元
   - 青髭補正強度: 青みを除去
3. **薄め具合を選択**（30%, 50%, 70%, 100%）
4. **「髭薄めを実行」をクリック**
5. **結果をギャラリーで確認**

#### Tab 3: 色調補正 + オーバーレイ

1. **「Tab 2の結果を取得」をクリック**
2. **補正モードを選択**
   - 青み除去（推奨）: LAB色空間で青髭を補正
   - 色味転送: 頬などの肌色を参照して補正
   - 自動補正: 周辺肌色を自動検出
3. **対象領域を指定**
   - Tab 1のマスクを使用
   - または手動で塗る
4. **髭オーバーレイ設定**（v5新機能）
   - 削除しなかった髭を元画像から重ねる
   - オーバーレイ強度とエッジぼかしを調整
5. **「色調補正 + オーバーレイを適用」をクリック**
6. **最終結果を確認・ダウンロード**

## パラメータ説明

### Tab 1: ルールベース検出

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 二値化閾値 | 80 | 小さいほど薄い髭も検出（暗い=髭） |
| 最小面積 | 10 | 検出領域の最小ピクセル数 |
| 最大面積 | 5000 | 検出領域の最大ピクセル数 |
| ノイズ除去 | OFF | 毛穴などの誤検出を防ぐバイラテラルフィルター |

### Tab 1: Grounded SAM（オプション）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 検出プロンプト | `beard. facial hair. stubble.` | 髭を表すテキスト |
| Box Threshold | 0.25 | ボックス検出の信頼度閾値 |
| Text Threshold | 0.20 | テキストマッチングの閾値 |

### Tab 2: MAT Inpainting

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| MAT強化モード | ON | テクスチャ保持 + 青髭補正 |
| テクスチャ強度 | 0.8 | 元画像の肌質感をどれだけ復元するか |
| 青髭補正強度 | 0.7 | 青みをどれだけ除去するか |

### Tab 3: 色調補正

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 補正強度 | 0.8 | 全体的な補正の強さ |
| a* 調整係数 | 0.3 | LAB色空間のa*チャンネル調整 |
| b* 調整係数 | 0.6 | LAB色空間のb*チャンネル調整 |
| L 調整係数 | 0.5 | LAB色空間の明度調整 |

### Tab 3: 髭オーバーレイ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| オーバーレイ強度 | 1.0 | 1.0 = 完全に元の髭、0.5 = 半透明 |
| エッジぼかし | 3 | 髭の境界をぼかして自然に馴染ませる |

### Tab 3: マスク隙間埋め

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 隙間埋めサイズ | 15 | 大きいほど広い隙間が埋まる（15-25推奨） |
| エッジぼかし | 21 | 大きいほど境界が滑らかに（21-31推奨） |

## トラブルシューティング

### LaMa が動作しない

```bash
pip install simple-lama-inpainting --no-deps
pip install numpy pillow opencv-python torch torchvision
```

### Grounded SAM の config が見つからない

`groundingdino-py` を pip でインストールすると、config ファイルは自動的にパッケージ内から取得されます。

```bash
pip install groundingdino-py
```

### CUDA メモリ不足

- 画像サイズを小さくする
- `config.py` の `MAX_IMAGE_SIZE` を調整

## 技術スタック

- **Gradio**: Web UI フレームワーク
- **OpenCV**: 画像処理（ルールベース検出、色空間変換）
- **MAT (Mask-Aware Transformer)**: 顔画像専用の高品質Inpainting（CVPR 2022）
- **SimpleLama**: LaMa ベースの汎用 Inpainting
- **LAB色空間**: 青髭補正（a*, b*チャンネル調整）
- **Segment Anything (SAM)**: セグメンテーション（オプション）
- **Grounding DINO**: テキストベース物体検出（オプション）

## 処理フロー

```
1. Tab 1: 髭検出 → 全検出マスク
2. Tab 1: 削除対象選択 → 削除マスク
3. Tab 2: 削除対象をInpainting (MAT/LaMa/OpenCV)
4. Tab 3: Inpainted画像を色補正（青み除去）
5. Tab 3: 残す髭マスク = 全検出 - 削除対象
6. Tab 3: 色補正画像 + 元画像の残す髭 = 最終出力
```

### 数式

```
remaining_beard = detect_mask - selection_mask
final_output = color_corrected * (1 - mask) + original * mask
```

## 参考文献

- [MAT Inpainting (Hugging Face)](https://huggingface.co/spacepxl/MAT-inpainting-fp16/blob/main/README.md) - MAT (Mask-Aware Transformer) のモデルとドキュメント

## ライセンス

MIT License
