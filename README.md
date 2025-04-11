# To Start
```
docker build . --no-cache
docker images # イメージIDの確認
docker run -it [イメージID] bash
```

# To Install
```
poetry install
```

# To Use
```
poetry run python main.py
```

# Memo About Tool
| ツール名       | 主な用途                         | 特徴・メリット                                                                 | 向いている場面                                |
|----------------|----------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------|
| **matplotlib** | グラフ作成                       | - 静的なグラフ画像として保存可能<br>- 高い柔軟性で論文向けの図表作成に最適        | 論文・スライド用の綺麗なグラフが欲しいとき     |
| **TensorBoard**| TensorFlow/PyTorch用の可視化UI | - ローカルでWeb UIが使える<br>- `add_scalar()`などで簡単にログ出力                | 学習中のlossやaccuracyをリアルタイムに見たい   |
| **MLflow**     | 実験トラッキングフレームワーク   | - 実験ごとにパラメータ・結果を記録<br>- Web UIあり<br>- 複数実験の比較が容易<br>- gitのcommitハッシュを記録して再現性を高める        | 学習条件を記録しながらモデルを比較したいとき   |