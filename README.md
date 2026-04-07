# 🎬 AutoSRT — 動画編集前処理自動化ツール

Premiere Pro 等での動画編集作業を効率化するための Web アプリケーションです。
ブラウザ上で **無音カット（FCP XML 生成）** と **LLM による SRT 字幕校正** を行えます。

---

## 🚀 機能

### フェーズ1: インテリジェント無音カット

- 音声ファイル（MP3/WAV）から無音区間を自動検出
- RMS ベースの高精度な無音判定
- パラメータ（音量閾値・最小無音時間・パディング）の調整が可能
- FCP XML を再構築して、無音区間を除外したタイムラインを生成
- Premiere Pro でそのまま読み込み可能なXMLを出力
- 波形ビジュアライゼーション付き

### フェーズ2: コンテキスト考慮型 SRT 校正

- LLM（Google Gemini）による高品質な字幕校正
- 動画のジャンルや文脈に基づいた「文脈認識型」校正
- 専門用語・固有名詞の正確な反映
- タイムスタンプを一切変更せず、テキストのみを校正
- 校正前後の差分プレビュー
- 長時間動画対応のチャンク分割処理

---

## 📦 セットアップ

### 前提条件

- Python 3.9 以上
- ffmpeg（MP3 ファイルを処理する場合に必要）

### インストール

```bash
# リポジトリをクローン / ディレクトリに移動
cd 20260407autosrt

# 仮想環境を作成（推奨）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

### ffmpeg のインストール（未インストールの場合）

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (chocolatey)
choco install ffmpeg
```

---

## 🖥️ 起動方法

```bash
streamlit run app.py
```

ブラウザが自動的に開き、アプリケーションが表示されます（デフォルト: `http://localhost:8501`）。

---

## 🔑 API キー設定（SRT 字幕校正機能に必要）

以下のいずれかの方法で設定してください。

### 方法1: アプリの UI（サイドバー）で入力

アプリ起動後、左サイドバーの「API キー」セクションに直接入力します。

### 方法2: 環境変数

```bash
export GEMINI_API_KEY="AIza..."
```

### 方法3: Streamlit Secrets

`.streamlit/secrets.toml` を作成:

```toml
GEMINI_API_KEY = "AIza..."
```

> **API キーの取得**: [Google AI Studio](https://aistudio.google.com/apikey) から無料で取得できます。

---

## 📂 ファイル構成

```
20260407autosrt/
├── app.py               # Streamlit メインアプリ（UI とルーティング）
├── audio_processor.py   # 音声解析と FCP XML 再構築のロジック
├── srt_processor.py     # SRT 解析と LLM 呼び出しのロジック
├── requirements.txt     # Python 依存パッケージ
└── README.md            # このファイル
```

---

## 💡 使用上の注意

### 無音カット機能

- 音声ファイルは動画から抽出したものをご使用ください（`ffmpeg -i video.mp4 -q:a 0 audio.mp3`）
- FCP XML は Premiere Pro の「ファイル > 書き出し > Final Cut Pro XML」から出力できます
- パラメータは動画の種類によって調整が必要です:
  - **トーク系動画**: 閾値 3〜8%、最小無音時間 300〜800ms
  - **ゲーム実況**: 閾値 5〜15%、最小無音時間 500〜1500ms
  - **音楽系**: 閾値 1〜3%、最小無音時間 1000〜3000ms
- パディングは 50〜200ms 程度を推奨（発話の頭切れ防止）

### SRT 字幕校正機能

- 文脈情報はできるだけ詳しく入力すると、校正精度が向上します
- 専門用語リストに音声認識で誤変換されやすい用語を含めてください
- チャンクサイズは 20〜50 程度が推奨です
- LLM のレート制限に注意してください（大量のチャンクを処理する場合）

---

## 🔧 技術詳細

### ppro Ticks 計算

Premiere Pro のタイムライン管理には `pproTicks`（1秒 = 254,016,000,000 ティック）が使用されます。
本アプリではこの値を正確に計算し、XML の `pproTicksIn` / `pproTicksOut` に反映します。

### 無音検出アルゴリズム

1. 音声データを 50ms ウィンドウで分割し、各ウィンドウの RMS 値を算出
2. 最大 RMS 値に対するパーセンテージで閾値を適用
3. 連続する無音ウィンドウを結合し、最小無音時間でフィルタリング
4. パディングを適用して有音区間を算出

### LLM 校正フロー

1. SRT ファイルをパースしてエントリ化
2. エントリをチャンク（デフォルト 30 エントリ）に分割
3. 各チャンクに対して、文脈情報を含むシステムプロンプトと共に LLM に送信
4. LLM 応答をパースし、タイムスタンプは元のものを維持（安全策）
5. 全チャンクを結合して新しい SRT ファイルを構築
