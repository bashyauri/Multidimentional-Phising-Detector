# FaceForensics++ Dataset

Place FaceForensics++ files here after your access request is approved.

Recommended local structure:

```text
datasets/faceforensics/
├── original/       # real/original videos or images
├── manipulated/    # fake/manipulated videos or images
└── frames/         # extracted training frames, if you generate them
```

Suggested label mapping:

- `original/` = legitimate/real = `0`
- `manipulated/` = deepfake/fake = `1`

FaceForensics++ is a licensed research dataset, so do not commit the downloaded
videos or extracted frames to Git. The project `.gitignore` keeps this folder
structure while ignoring the heavy media files.

For an MSc-sized experiment, use a manageable subset first, for example:

- 100 to 500 original videos or sampled frames
- 100 to 500 manipulated videos or sampled frames
- an 80/20 train/test split

After the data is in this folder, the deepfake training script can read from
`original/` and `manipulated/` and save a trained model in `models/`.
