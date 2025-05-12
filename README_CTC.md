# Connectionist Temporal Classification (CTC) Loss Implementation

This project now supports CTC loss for training captcha recognition models. CTC loss is particularly well-suited for sequence recognition tasks like captcha solving as it doesn't require precise alignment between input and output sequences.

## Benefits of CTC Loss

- Doesn't require precise character-by-character alignment
- Can handle variable-length sequences better
- Often improves recognition of connected or overlapping characters
- Allows the model to learn the alignment between input and output sequences

## Implementation Details

The CTC loss has been implemented as an option in the existing model architecture. Key changes include:

1. Added a blank token for CTC calculations (using CLASS_NUM as the blank index)
2. Modified the model output processing to work with CTC format
3. Added command-line arguments to enable/disable CTC loss
4. Updated experiment naming and logging

## How to Use

### Training with CTC Loss

To train a model using CTC loss, simply add the `--use_ctc` flag:

```bash
python launcher.py --model_name resnet --use_ctc
```

### Testing with CTC Loss

When testing a model trained with CTC loss, make sure to use the same flag:

```bash
python test.py --use_ctc --ckpt /path/to/your/checkpoint.ckpt
```

### Prediction with CTC Loss

For making predictions with a model trained using CTC loss:

```bash
python predictor.py --use_ctc --ckpt /path/to/your/checkpoint.ckpt --input /path/to/captcha/image.png
```

## Comparing with Standard Loss

You may want to compare the performance of models trained with and without CTC loss:

1. Train two separate models (one with CTC, one without)
2. Test both models on the same validation/test set
3. Compare accuracy metrics and prediction results

The model with CTC loss may perform better on more complex captchas with overlapping or connected characters. 
