# Mac MPSで実装するために一部処理(torchvision.ops.roi_align)のためにCPUにFallbackするための環境変数を設定
export PYTORCH_ENABLE_MPS_FALLBACK=1

# testデータでCountingDINOモデルを実装
# 注意点1: training, valdationデータの場合は、--split引数にそれぞれt--split train --split valを指定 
# 注意点2: DINO ViT L/14 reg.モデル(--model_name dinov2_vitl14_reg)は対規模モデル過ぎてMPSでも処理しきれないため、モデルを軽量化する
# 注意点3: --divide_et_impera_twice Trueにすると、解像度がオリジナルの４倍になり、特徴抽出においてメモリ使用量が爆発する。
  # 理論： Google社のVision Transformer(ViT)では840*840の画像サイズにしか対応していないため、それ以上の高解像度処理にはオリジナル画像を複数の”パッチ”に分割して、それぞれで特徴抽出することで解像度をあげる。各パッチにて類似性マッチング／畳み込みを行い、最後にDensity Mapで結合する.
  # --divide_et_imperaであれば、解像度がオリジナルの２倍になるが、４倍ほどの処理量にはならないため、Trueで良い
python convolutional_counting.py \
  --model_name dinov2_vitl14_reg \
  --divide_et_impera True \
  --divide_et_impera_twice False \
  --filter_background True \
  --ellipse_normalization True \
  --ellipse_kernel_cleaning True \
  --split test \
  --img_dir /Users/isaoishikawa/FSC147_384_V2/images_384_VarV2 \
  --density_map_dir /Users/isaoishikawa/FSC147_384_V2/gt_density_map_adaptive_384_VarV2 \
  --annotation /Users/isaoishikawa/FSC147_384_V2/annotation_FSC147_384.json \
  --splits /Users/isaoishikawa/FSC147_384_V2/Train_Test_Val_FSC_147.json \
  --no_skip True

# TODO⭐️100枚画像でトレーニングしたモデルで精度確認 (train, val, test)
# TODO⭐️任意の画像1枚に対してのカウント結果を出力
# 学習した
python convolutional_counting.py \
  --model_name dinov2_vitl14_reg \
  --divide_et_impera True \
  --divide_et_impera_twice False \
  --filter_background True \
  --ellipse_normalization True \
  --ellipse_kernel_cleaning True \
  --split test \
  --img_dir /Users/isaoishikawa/my_custom_data/images_100 \
  --density_map_dir /Users/isaoishikawa/my_custom_data/gt_density_map_adaptive_512_512_object_VarV2 \
  --annotation /Users/isaoishikawa/my_custom_data/annotation_my100.json \
  --splits /Users/isaoishikawa/my_custom_data/Train_Test_Val_my100.json \
  --no_skip True