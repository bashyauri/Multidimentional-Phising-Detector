# Evaluation Data Guide

This project is a multimodal phishing detection prototype. The app can scan URL,
email, SMS, QR, video deepfake, and voice deepfake inputs, but the quality of the
defence depends on having clear labelled evaluation samples.

## What To Show In Defence

Show evidence for four things:

1. The dataset source and labels.
2. The trained model files.
3. The saved metrics and confusion matrices.
4. Live predictions from labelled evaluation samples.

Recommended evidence files:

```text
models/metrics_summary.json
models/url_metrics.json
models/email_metrics.json
models/sms_metrics.json
models/deepfake_efficientnet_b0_metrics.json
models/voice_deepfake_metrics_balanced.json

static/plots/url_confusion_matrix.png
static/plots/email_confusion_matrix.png
static/plots/sms_confusion_matrix.png
static/plots/deepfake_confusion_matrix.png
static/plots/voice_confusion_matrix.png
```

## Recommended Evaluation Folder

Create a small separate evaluation pack:

```text
datasets/evaluation_samples/
├── url/
│   ├── legitimate_urls.csv
│   └── phishing_urls.csv
├── email/
│   ├── legitimate_email_samples.csv
│   └── phishing_email_samples.csv
├── sms/
│   ├── legitimate_sms_samples.csv
│   └── phishing_sms_samples.csv
├── qr/
│   ├── legitimate/
│   └── phishing/
├── video_deepfake/
│   ├── real/
│   └── fake/
└── voice_deepfake/
    ├── real/
    └── fake/
```

Keep this evaluation pack separate from training data where possible.

## Required Samples Per Module

| Module | Minimum Demo Samples | Better Defence Samples | Label Meaning |
|---|---:|---:|---|
| URL | 20 legitimate + 20 phishing | 50 + 50 | phishing vs legitimate |
| Email | 20 legitimate + 20 phishing | 50 + 50 | phishing vs legitimate |
| SMS | 20 ham + 20 spam | 50 + 50 | spam/smishing vs ham |
| QR | 10 legitimate + 10 phishing | 20 + 20 | decoded URL is scored |
| Video deepfake | 10 real + 10 fake | 20 + 20 | fake = manipulated |
| Voice deepfake | 10 real + 10 fake | 20 + 20 | fake = synthetic/spoofed voice |

## Deepfake Video Size Problem

FaceForensics++ videos can be large. This is normal. Do not try to include a
huge video dataset in the project repository or client demo.

For defence, use a small compressed subset:

```text
datasets/evaluation_samples/video_deepfake/real/
datasets/evaluation_samples/video_deepfake/fake/
```

Use 10 to 20 real videos and 10 to 20 fake videos.

### Compress Videos

Use FFmpeg to create short, small clips:

```powershell
ffmpeg -i input.mp4 -t 5 -vf "scale=320:-1" -c:v libx264 -crf 28 -an output.mp4
```

Meaning:

- `-t 5` keeps only the first 5 seconds.
- `scale=320:-1` reduces video width to 320 pixels.
- `-crf 28` compresses the video.
- `-an` removes audio because video deepfake detection does not need it.

Example fake clip:

```powershell
ffmpeg -i datasets\faceforensics\manipulated\000_003.mp4 -t 5 -vf "scale=320:-1" -c:v libx264 -crf 28 -an datasets\evaluation_samples\video_deepfake\fake\fake_01.mp4
```

Example real clip:

```powershell
ffmpeg -i datasets\faceforensics\original\000.mp4 -t 5 -vf "scale=320:-1" -c:v libx264 -crf 28 -an datasets\evaluation_samples\video_deepfake\real\real_01.mp4
```

### Even Smaller Option: Use Images

The app accepts images for deepfake scan. Extract one frame per video:

```powershell
ffmpeg -i input.mp4 -ss 00:00:02 -vframes 1 output.jpg
```

Example:

```powershell
ffmpeg -i datasets\faceforensics\manipulated\000_003.mp4 -ss 00:00:02 -vframes 1 datasets\evaluation_samples\video_deepfake\fake\fake_01.jpg
```

This is useful when storage is limited.

## Voice Deepfake Evaluation

The voice detector is trained for short speech clips, not music.

Use:

```text
16 kHz mono WAV
3 to 5 seconds
speech only
```

Avoid:

```text
full songs
instrumental audio
background music
long MP3 files
noisy non-speech clips
```

If you have an MP3 speech clip, convert it:

```powershell
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

The project also has synthetic upload-test fixtures:

```text
datasets/voice/test_samples/legitimate/
datasets/voice/test_samples/fake/
```

Generate them with:

```text
generate_voice_test_samples.bat
```

These are only for checking the endpoint. Do not use them as research
evaluation data.

## Batch Helpers

The project includes two helper batch files:

```text
generate_voice_test_samples.bat
generate_video_deepfake_eval_samples.bat
generate_voice_eval_samples.bat
generate_text_qr_eval_samples.bat
```

Voice helper:

```powershell
generate_voice_test_samples.bat
```

Video helper, create 10 real and 10 fake compressed clips:

```powershell
generate_video_deepfake_eval_samples.bat
```

Video helper, create 20 real and 20 fake JPEG frames:

```powershell
generate_video_deepfake_eval_samples.bat 20 frame
```

Output:

```text
datasets/evaluation_samples/video_deepfake/real/
datasets/evaluation_samples/video_deepfake/fake/
```

Voice evaluation helper, create 10 real and 10 fake samples:

```powershell
generate_voice_eval_samples.bat
```

Voice evaluation helper, create 20 real and 20 fake samples:

```powershell
generate_voice_eval_samples.bat 20
```

Output:

```text
datasets/evaluation_samples/voice_deepfake/real/
datasets/evaluation_samples/voice_deepfake/fake/
```

The helper first looks for real evaluation audio in:

```text
datasets/voice/evaluation/real/
datasets/voice/real/
datasets/voice/test_samples/legitimate/
```

It looks for fake evaluation audio in:

```text
datasets/voice/evaluation/fake/
datasets/voice/fake/
datasets/voice/test_samples/fake/
```

If only the synthetic test samples exist, it will copy those. For final MSc
evaluation, replace them with real labelled speech samples where possible.

Text and QR helper:

```powershell
generate_text_qr_eval_samples.bat
```

Output:

```text
datasets/evaluation_samples/url/legitimate_urls.csv
datasets/evaluation_samples/url/phishing_urls.csv
datasets/evaluation_samples/sms/legitimate_sms_samples.csv
datasets/evaluation_samples/sms/phishing_sms_samples.csv
datasets/evaluation_samples/email/legitimate_email_samples.csv
datasets/evaluation_samples/email/phishing_email_samples.csv
datasets/evaluation_samples/qr/legitimate/
datasets/evaluation_samples/qr/phishing/
```

The QR images require the Python `qrcode` package. If QR generation is skipped,
install it with:

```powershell
pip install qrcode[pil]
```

## How To Report Results

Create a simple table for the client or supervisor:

| Module | Samples | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| URL | 40 | 38 | 2 | 95% |
| Email | 40 | 39 | 1 | 97.5% |
| SMS | 40 | 37 | 3 | 92.5% |
| QR | 20 | 18 | 2 | 90% |
| Video deepfake | 40 | 32 | 8 | 80% |
| Voice deepfake | 40 | 36 | 4 | 90% |

Use your actual results after testing.

## Important Defence Wording

Recommended wording:

> Due to the large size of full multimedia datasets such as FaceForensics++, a
> compressed representative evaluation subset was created for demonstration and
> testing. The full dataset source is documented, while the application
> evaluation uses labelled real and manipulated samples.

For voice:

> The voice deepfake module is designed for short speech samples. It is not a
> music or general audio classifier.

For the whole app:

> The system is a research prototype demonstrating multimodal phishing detection
> across text, URL, QR, video, and voice signals. It is not yet a production
> forensic system.
