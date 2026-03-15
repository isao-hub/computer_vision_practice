#!/usr/bin/env python3
"""
色調補正ツール（肌色調整用）

使い方:
1. 「対象領域」モードで色を調整したい領域（髭など）をマウスで塗る
2. 「スポイト」モードで目標の色（頬など）の領域をマウスで塗る
3. 「適用」ボタンで対象領域の色味をスポイト領域に合わせる
   （明るさ・テクスチャは保持され、色相・彩度のみ調整）
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import sys


class ColorFillTool:
    def __init__(self, root):
        self.root = root
        self.root.title("肌色調整ツール")

        # 画像データ
        self.original_image = None
        self.display_image = None
        self.result_image = None

        # マスク（塗りつぶし領域）
        self.target_mask = None  # 塗りつぶす対象の領域
        self.color_mask = None   # 色を取得する領域

        # モード: 'target' または 'color'
        self.current_mode = tk.StringVar(value='target')

        # ブラシサイズ
        self.brush_size = tk.IntVar(value=15)

        # 色調補正の強度（0.0〜1.0）
        self.blend_strength = tk.DoubleVar(value=1.0)

        # マウス状態
        self.drawing = False
        self.last_point = None

        # スケール係数
        self.scale = 1.0

        self._setup_ui()

    def _setup_ui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # コントロールパネル（左側）
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        control_frame.pack_propagate(False)

        # 画像読み込みボタン
        ttk.Button(control_frame, text="画像を開く", command=self._open_image).pack(fill=tk.X, pady=5)

        # モード選択
        mode_frame = ttk.LabelFrame(control_frame, text="モード選択")
        mode_frame.pack(fill=tk.X, pady=10)

        self.target_radio = ttk.Radiobutton(
            mode_frame,
            text="🎯 対象領域（髭など）",
            variable=self.current_mode,
            value='target',
            command=self._on_mode_change
        )
        self.target_radio.pack(anchor=tk.W, padx=5, pady=5)

        self.color_radio = ttk.Radiobutton(
            mode_frame,
            text="💧 スポイト（頬など）",
            variable=self.current_mode,
            value='color',
            command=self._on_mode_change
        )
        self.color_radio.pack(anchor=tk.W, padx=5, pady=5)

        # ブラシサイズ
        brush_frame = ttk.LabelFrame(control_frame, text="ブラシサイズ")
        brush_frame.pack(fill=tk.X, pady=10)

        self.brush_scale = ttk.Scale(
            brush_frame,
            from_=1,
            to=100,
            variable=self.brush_size,
            orient=tk.HORIZONTAL
        )
        self.brush_scale.pack(fill=tk.X, padx=5, pady=5)

        self.brush_label = ttk.Label(brush_frame, text=f"サイズ: {self.brush_size.get()}")
        self.brush_label.pack(padx=5)
        self.brush_size.trace_add('write', self._update_brush_label)

        # 色調補正の強度
        strength_frame = ttk.LabelFrame(control_frame, text="色調補正の強度")
        strength_frame.pack(fill=tk.X, pady=10)

        self.strength_scale = ttk.Scale(
            strength_frame,
            from_=0.0,
            to=1.0,
            variable=self.blend_strength,
            orient=tk.HORIZONTAL
        )
        self.strength_scale.pack(fill=tk.X, padx=5, pady=5)

        self.strength_label = ttk.Label(strength_frame, text="強度: 100%")
        self.strength_label.pack(padx=5)
        self.blend_strength.trace_add('write', self._update_strength_label)

        # 操作ボタン
        action_frame = ttk.LabelFrame(control_frame, text="操作")
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="✅ 適用", command=self._apply_fill).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="🔄 マスクをクリア", command=self._clear_masks).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="↩️ 元に戻す", command=self._reset_image).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="💾 保存", command=self._save_image).pack(fill=tk.X, padx=5, pady=5)

        # 色プレビュー
        preview_frame = ttk.LabelFrame(control_frame, text="目標色（スポイト平均）")
        preview_frame.pack(fill=tk.X, pady=10)

        self.color_preview = tk.Canvas(preview_frame, width=180, height=50, bg='gray')
        self.color_preview.pack(padx=5, pady=5)

        self.color_label = ttk.Label(preview_frame, text="R:-- G:-- B:--")
        self.color_label.pack(padx=5, pady=5)

        # ステータス
        self.status_label = ttk.Label(control_frame, text="画像を開いてください", wraplength=180)
        self.status_label.pack(fill=tk.X, pady=10)

        # キャンバス（右側）
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # スクロールバー付きキャンバス
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            canvas_frame,
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set,
            bg='gray'
        )

        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)

        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # マウスイベント
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)  # macOS/Windows
        self.canvas.bind('<Button-4>', self._on_mousewheel)    # Linux scroll up
        self.canvas.bind('<Button-5>', self._on_mousewheel)    # Linux scroll down

    def _update_brush_label(self, *args):
        self.brush_label.config(text=f"サイズ: {self.brush_size.get()}")

    def _update_strength_label(self, *args):
        strength = int(self.blend_strength.get() * 100)
        self.strength_label.config(text=f"強度: {strength}%")

    def _on_mode_change(self):
        mode = self.current_mode.get()
        if mode == 'target':
            self.status_label.config(text="対象領域モード：色を調整したい領域（髭など）を塗ってください")
        else:
            self.status_label.config(text="スポイトモード：目標の色の領域（頬など）を塗ってください")

    def _open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._load_image(file_path)

    def _load_image(self, path):
        # OpenCVで画像読み込み（BGR）
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            self.status_label.config(text="画像の読み込みに失敗しました")
            return

        self.result_image = self.original_image.copy()

        # マスクを初期化
        h, w = self.original_image.shape[:2]
        self.target_mask = np.zeros((h, w), dtype=np.uint8)
        self.color_mask = np.zeros((h, w), dtype=np.uint8)

        # スケールを計算
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            self.scale = min(canvas_w / w, canvas_h / h, 1.0)
        else:
            self.scale = 1.0

        self._update_display()
        self.status_label.config(text=f"画像を読み込みました ({w}x{h})")
        self._on_mode_change()

    def _update_display(self):
        if self.result_image is None:
            return

        # 結果画像をコピー
        display = self.result_image.copy()

        # マスクをオーバーレイ表示
        # 対象領域：赤で半透明表示
        if self.target_mask is not None:
            target_overlay = np.zeros_like(display)
            target_overlay[:, :, 2] = 255  # 赤
            display = np.where(
                self.target_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(display, 0.5, target_overlay, 0.5, 0),
                display
            )

        # スポイト領域：緑で半透明表示
        if self.color_mask is not None:
            color_overlay = np.zeros_like(display)
            color_overlay[:, :, 1] = 255  # 緑
            display = np.where(
                self.color_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(display, 0.5, color_overlay, 0.5, 0),
                display
            )

        # BGR -> RGB変換
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        # スケーリング
        h, w = display_rgb.shape[:2]
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        if self.scale != 1.0:
            display_rgb = cv2.resize(display_rgb, (new_w, new_h))

        # PIL画像に変換
        pil_image = Image.fromarray(display_rgb)
        self.display_image = ImageTk.PhotoImage(pil_image)

        # キャンバスを更新
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        # スポイト領域の平均色を更新
        self._update_color_preview()

    def _update_color_preview(self):
        if self.result_image is None or self.color_mask is None:
            return

        # スポイト領域の平均色を計算
        if np.any(self.color_mask > 0):
            mean_color = cv2.mean(self.result_image, mask=self.color_mask)[:3]
            b, g, r = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])

            # プレビューを更新
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            self.color_preview.config(bg=hex_color)
            self.color_label.config(text=f"R:{r} G:{g} B:{b}")
        else:
            self.color_preview.config(bg='gray')
            self.color_label.config(text="R:-- G:-- B:--")

    def _get_canvas_coords(self, event):
        # キャンバス上の座標を画像座標に変換
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # スケールを考慮して実際の画像座標に変換
        img_x = int(canvas_x / self.scale)
        img_y = int(canvas_y / self.scale)

        return img_x, img_y

    def _on_mouse_down(self, event):
        if self.result_image is None:
            return

        self.drawing = True
        x, y = self._get_canvas_coords(event)
        self.last_point = (x, y)
        self._draw_at(x, y)

    def _on_mouse_drag(self, event):
        if not self.drawing or self.result_image is None:
            return

        x, y = self._get_canvas_coords(event)

        # 前の点から現在の点まで線を引く
        if self.last_point is not None:
            self._draw_line(self.last_point, (x, y))

        self.last_point = (x, y)
        self._update_display()

    def _on_mouse_up(self, event):
        self.drawing = False
        self.last_point = None

    def _draw_at(self, x, y):
        if self.result_image is None:
            return

        h, w = self.result_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            mask = self.target_mask if self.current_mode.get() == 'target' else self.color_mask
            cv2.circle(mask, (x, y), self.brush_size.get(), 255, -1)
            self._update_display()

    def _draw_line(self, pt1, pt2):
        if self.result_image is None:
            return

        mask = self.target_mask if self.current_mode.get() == 'target' else self.color_mask
        cv2.line(mask, pt1, pt2, 255, self.brush_size.get() * 2)

    def _on_mousewheel(self, event):
        # マウスホイールでブラシサイズ変更
        if event.num == 4 or event.delta > 0:
            self.brush_size.set(min(100, self.brush_size.get() + 2))
        elif event.num == 5 or event.delta < 0:
            self.brush_size.set(max(1, self.brush_size.get() - 2))

    def _apply_fill(self):
        if self.result_image is None:
            self.status_label.config(text="画像を開いてください")
            return

        if not np.any(self.target_mask > 0):
            self.status_label.config(text="対象領域を選択してください")
            return

        if not np.any(self.color_mask > 0):
            self.status_label.config(text="スポイト領域を選択してください")
            return

        # LAB色空間に変換
        lab_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # スポイト領域のピクセルを取得
        color_pixels_lab = lab_image[self.color_mask > 0]

        # 明度(L)ごとの色味(a*, b*)のルックアップテーブルを作成
        # L値は0-255の範囲
        l_to_a = np.zeros(256, dtype=np.float32)
        l_to_b = np.zeros(256, dtype=np.float32)
        l_counts = np.zeros(256, dtype=np.float32)

        # スポイト領域のピクセルからL→(a,b)のマッピングを作成
        for pixel in color_pixels_lab:
            l_idx = int(pixel[0])
            l_to_a[l_idx] += pixel[1]
            l_to_b[l_idx] += pixel[2]
            l_counts[l_idx] += 1

        # 平均を計算（データがある部分のみ）
        valid = l_counts > 0
        l_to_a[valid] /= l_counts[valid]
        l_to_b[valid] /= l_counts[valid]

        # データがない明度レベルを補間で埋める
        l_to_a = self._interpolate_lut(l_to_a, l_counts)
        l_to_b = self._interpolate_lut(l_to_b, l_counts)

        # LUTをスムージング（ガウシアンフィルタ）
        kernel_size = 21  # スムージングの強さ
        l_to_a = cv2.GaussianBlur(l_to_a.reshape(1, -1), (kernel_size, 1), 0).flatten()
        l_to_b = cv2.GaussianBlur(l_to_b.reshape(1, -1), (kernel_size, 1), 0).flatten()

        # 強度
        strength = self.blend_strength.get()

        # 対象領域の各ピクセルを処理
        target_coords = np.where(self.target_mask > 0)

        for y, x in zip(target_coords[0], target_coords[1]):
            l_val = int(lab_image[y, x, 0])
            original_a = lab_image[y, x, 1]
            original_b = lab_image[y, x, 2]

            # 目標の色味を取得
            target_a = l_to_a[l_val]
            target_b = l_to_b[l_val]

            # 強度に応じてブレンド
            lab_image[y, x, 1] = original_a + (target_a - original_a) * strength
            lab_image[y, x, 2] = original_b + (target_b - original_b) * strength

        # 値を0-255の範囲にクリップ
        lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)

        # エッジを滑らかにするためにマスク境界をぼかす
        # マスクされた領域と元の画像をブレンド
        blur_mask = cv2.GaussianBlur(self.target_mask.astype(np.float32), (15, 15), 0)
        blur_mask = blur_mask / 255.0

        # BGRに戻す
        result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # 元の画像とブレンド（境界を滑らかに）
        for c in range(3):
            self.result_image[:, :, c] = (
                self.result_image[:, :, c] * (1 - blur_mask) +
                result_bgr[:, :, c] * blur_mask
            ).astype(np.uint8)

        # マスクをクリア
        self.target_mask.fill(0)
        self.color_mask.fill(0)

        self._update_display()
        self.status_label.config(text="色調補正を適用しました")

    def _interpolate_lut(self, lut, counts):
        """データがない明度レベルを線形補間で埋める"""
        result = lut.copy()
        valid_indices = np.where(counts > 0)[0]

        if len(valid_indices) == 0:
            return result

        # 最初と最後の有効値で端を埋める
        if valid_indices[0] > 0:
            result[:valid_indices[0]] = result[valid_indices[0]]
        if valid_indices[-1] < 255:
            result[valid_indices[-1]+1:] = result[valid_indices[-1]]

        # 中間の欠損値を線形補間
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            if end_idx - start_idx > 1:
                # 線形補間
                start_val = result[start_idx]
                end_val = result[end_idx]
                for j in range(start_idx + 1, end_idx):
                    t = (j - start_idx) / (end_idx - start_idx)
                    result[j] = start_val + (end_val - start_val) * t

        return result

    def _clear_masks(self):
        if self.target_mask is not None:
            self.target_mask.fill(0)
        if self.color_mask is not None:
            self.color_mask.fill(0)
        self._update_display()
        self.status_label.config(text="マスクをクリアしました")

    def _reset_image(self):
        if self.original_image is not None:
            self.result_image = self.original_image.copy()
            self._clear_masks()
            self.status_label.config(text="画像を元に戻しました")

    def _save_image(self):
        if self.result_image is None:
            self.status_label.config(text="保存する画像がありません")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            cv2.imwrite(file_path, self.result_image)
            self.status_label.config(text=f"保存しました: {file_path}")


def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = ColorFillTool(root)

    # コマンドライン引数から画像を読み込む
    if len(sys.argv) > 1:
        app._load_image(sys.argv[1])

    root.mainloop()


if __name__ == '__main__':
    main()
