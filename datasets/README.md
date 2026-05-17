# Datasets

Place your research datasets in this folder.

Expected default filenames:
- `url_phishing.csv`
- `email_phishing.csv`
- `sms_spam.csv`
- `faceforensics/`

## Where to get datasets

Use any academically acceptable public datasets, then export to CSV.

Direct dataset links:

- URL phishing:
	- https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
	- https://archive.ics.uci.edu/dataset/327/phishing+websites

- Email phishing:
	- https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
	- https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

- SMS phishing/spam:
	- https://archive.ics.uci.edu/dataset/228/sms+spam+collection

After download:
1. Keep only the columns needed for training (text/url + label).
2. Save each as CSV.
3. Rename to the expected default filenames above.
4. Place all CSV files in this `datasets/` folder.
5. Run `validate_datasets.bat` from project root to verify files before training.

Important:
- Kaggle downloads require login.
- If files are .arff/.txt/.zip, extract and convert to CSV first.

## QR and Deepfake data requirements

- QR module: no dedicated training dataset needed.
	Upload QR image files in the web app. The extracted URL is scored by the URL model.

- Deepfake module (simplified): no training dataset required in this prototype.
	The module returns a simulated score for research workflow demonstration.

- Deepfake module (real dataset option): place FaceForensics++ media inside
	`datasets/faceforensics/`.
	Use `original/` for real samples and `manipulated/` for fake samples.
	The actual media files are ignored by Git because FaceForensics++ is large
	and distributed under dataset access terms.

## Required columns

### URL dataset
- URL column candidates: `url`, `domain`, `link`
- Label column candidates: `label`, `result`, `class`, `target`, `status`

### Email dataset
- Text column candidates: `text`, `email`, `content`, `message`, `body`
- Label column candidates: `label`, `class`, `target`, `result`

### SMS dataset
- Text column candidates: `text`, `sms`, `message`, `content`
- Label column candidates: `label`, `class`, `target`, `result`

Label values are normalized automatically. Typical values supported include:
- Phishing class: `1`, `phishing`, `spam`, `malicious`
- Legitimate class: `0`, `legitimate`, `ham`, `safe`
