# 🚀 AutoSRT 導入ガイド（他の人向け）

このフォルダを受け取った方が、自分のパソコンで AutoSRT を動かすための手順書です。

## 1. 準備するもの

アプリを動かす前に、以下の3つを準備してください。

### ① Python のインストール
Python 3.9 以降が必要です。[公式サイト](https://www.python.org/downloads/) からインストールしてください。
※インストール時、「Add Python to PATH」にチェックを入れるのを忘れないでください。

### ② FFmpeg のインストール
音声処理に必須のツールです。
- **Mac:** ターミナルで `brew install ffmpeg` を実行。
- **Windows:** [こちらのサイト](https://github.com/BtbN/FFmpeg-Builds/releases) 等から `ffmpeg-master-latest-win64-gpl.zip` をダウンロードし、解凍して `bin` フォルダにパスを通してください。

### ③ Gemini API キー
[Google AI Studio](https://aistudio.google.com/apikey) から無料で取得できます。

---

## 2. 使い方（初回のみ）

フォルダを解凍し、ターミナル（WindowsはコマンドプロンプトやPowerShell）を開いて以下のコマンドを実行します。

```bash
# 必要なライブラリをまとめてインストール
pip install -r requirements.txt
```

---

## 3. アプリの起動

準備ができたら、毎回以下のコマンドでアプリを起動します。

```bash
streamlit run app.py
```

実行後、ブラウザが立ち上がりアプリ画面が表示されます。
左側の設定パネルに取得した **Gemini API キー** を入力して使用を開始してください。

---

## 📂 フォルダの中身
- `app.py`: アプリのメイン画面
- `audio_processor.py`: 無音カットの計算ロジック
- `srt_processor.py`: AI字幕修正のロジック
- `requirements.txt`: 必要なライブラリ一覧
