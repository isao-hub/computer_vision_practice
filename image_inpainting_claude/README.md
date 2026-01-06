# Progressive Image Inpainting

Google「消しゴムマジック」のような機能を持つ画像修復アプリケーション。物体を段階的に削除し、そのプロセスを可視化します。

## Clude Codeへのインプット

画像内の物体を修復するアプリを開発したいと思っています。Google社の「消しゴムマジック」のような機能を実装したいです。利用する技術はPython, simple_lama_inpaintingです。主なユースケースは、いくつかのシチュエーションにおいて画像内の物体が段階的に消えていく加工ができるアプリだと嬉しい。simple_lama_inpaintingに関する技術アルゴリズムの情報も渡すから参照して。\
情報1:https://github.com/enesmsahin/simple-lama-inpainting\
情報2:https://github.com/advimman/lama

## 特徴

- **段階的な物体削除**: 20%, 40%, 60%, 80%, 100%のステージで徐々に物体を削除
- **高品質な修復**: LaMa (Large Mask Inpainting) アルゴリズムを使用したフーリエ畳み込みベースの画像修復
- **直感的なUI**: Gradioベースのインタラクティブなウェブインターフェース
- **手動マスク描画**: 削除したい物体を自由に選択可能

## インストール

### 前提条件

- Python 3.8以上
- pip

### セットアップ

1. リポジトリをクローン（または展開）:
```bash
cd image_inpainting_claude
```

2. 依存パッケージをインストール:
```bash
pip install -r requirements.txt
```

## 使い方

### アプリケーションの起動

```bash
python app.py
```

アプリケーションが起動すると、ブラウザで `http://localhost:7860` が開きます。

### 操作手順

1. **画像をアップロード**: 物体を削除したい画像を選択
2. **マスクを描画**: ImageEditorで削除したい物体を白で塗りつぶす
3. **マスクを追加**: 描画したマスクを保存（各マスク = 1つの物体）
4. **繰り返し**: 削除したい物体ごとに手順2-3を繰り返す
5. **ステージ選択**: 生成したいステージ（%）を選択
6. **実行**: 「段階的削除を実行」ボタンをクリック

### 例

10個の物体（マスク）がある場合:
- **20%ステージ**: 2個削除（8個残存）
- **40%ステージ**: 4個削除（6個残存）
- **60%ステージ**: 6個削除（4個残存）
- **80%ステージ**: 8個削除（2個残存）
- **100%ステージ**: 全削除

## プロジェクト構造

```
claude_code/
├── app.py                      # メインアプリケーション
├── requirements.txt            # 依存パッケージ
├── config.py                   # 設定定数
├── core/
│   ├── __init__.py
│   ├── image_utils.py         # 画像処理ユーティリティ
│   ├── mask_processor.py      # マスク集約とステージ計算
│   └── inpainting.py          # SimpleLamaラッパー
├── ui/
│   ├── __init__.py
│   ├── components.py          # Gradio UIコンポーネント
│   └── callbacks.py           # イベントハンドラー
└── examples/
    └── sample_image.jpg       # サンプル画像
```

## 技術スタック

- **Python**: バックエンド処理
- **Gradio 4.x**: Webインターフェース
- **simple-lama-inpainting**: 画像修復エンジン
- **LaMa**: フーリエ畳み込みベースの inpainting アルゴリズム
- **PIL/Pillow**: 画像処理
- **NumPy**: 配列操作
- **OpenCV**: 追加の画像処理

## アーキテクチャ

### 処理フロー

1. **マスク収集**: ユーザーが複数のマスクを描画・追加
2. **ステージ計算**: 選択されたパーセンテージに基づいてマスク数を計算
3. **累積マスク生成**: 各ステージに対して累積マスクを生成
4. **Inpainting実行**: 各ステージを元画像から独立して処理
5. **結果表示**: ギャラリー形式で段階的な結果を表示

### 主要な設計決定

#### 1. Sequential Mask Addition
Gradioの制限により、ユーザーが1つずつマスクを追加する方式を採用。`gr.State`で累積マスクリストを管理。

#### 2. Cumulative Processing
各ステージを元画像から独立して処理することで、inpainting品質の劣化を防止。

#### 3. Fixed Stage Percentages
シンプルさを優先し、デフォルトで20%, 40%, 60%, 80%, 100%の5ステージを設定。

## パフォーマンス

- **画像サイズ制限**: 自動的に最大2048pxにリサイズ（メモリ節約）
- **処理時間** (概算):
  - 1ステージ: 2-5秒（CPU）、0.5-1秒（GPU）
  - 5ステージ: 10-25秒（CPU）、2.5-5秒（GPU）
- **最適化**: CUDA対応GPUがあれば自動的に使用

## トラブルシューティング

### SimpleLamaのインストールエラー

PyTorchが正しくインストールされているか確認してください:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

GPU版が必要な場合:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### メモリ不足エラー

- より小さい画像を使用する
- 処理するマスク数を減らす
- ステージ数を減らす

## 将来の拡張機能

- [ ] アニメーションGIF出力
- [ ] マスクのUndo/Redo機能
- [ ] マスク順序のドラッグ&ドロップ変更
- [ ] カスタムステージパーセンテージ
- [ ] バッチ処理（複数画像）
- [ ] マスク設定の保存/読み込み
- [ ] 代替inpaintingモデルのサポート

## 参考文献

- [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) - Python wrapper for LaMa inpainting
- [LaMa](https://github.com/advimman/lama) - Resolution-robust Large Mask Inpainting with Fourier Convolutions
- [Gradio](https://gradio.app/) - Build ML web apps in Python

## ライセンス

このプロジェクトは教育目的で作成されました。
使用しているライブラリのライセンスを確認してください:
- simple-lama-inpainting: Apache-2.0
- Gradio: Apache-2.0

## 貢献

バグ報告や機能リクエストは歓迎します。

## 作者

Claude Code assisted implementation
