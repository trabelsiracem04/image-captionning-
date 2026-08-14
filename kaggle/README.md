# Train stage-1 on Kaggle

Two Kaggle **Datasets** (uploaded once) + one **Notebook** (created on kaggle.com, 1 pasted cell). No code is copy-pasted into cells by hand.

## 1. Build the two zips

From the repo root (on your machine):

```bash
python scripts/make_kaggle_bundle.py --code   # -> dist/kaggle_code.zip  (~0.02 MiB)
python scripts/make_kaggle_bundle.py --data   # -> dist/kaggle_data.zip  (~4.1 GB)
```

## 2. Upload them as Kaggle Datasets

1. kaggle.com -> **Datasets -> New Dataset**.
2. Drag `dist/kaggle_code.zip`. Name it `image-captioning-code`. Create.
3. Repeat for `dist/kaggle_data.zip`, name it `image-captioning-data`.

> The zip names can differ from the dataset slugs; the notebook uses the **dataset slug** (`image-captioning-code` / `image-captioning-data`), so name them exactly as above, or edit the two `SLUG` vars in the notebook cell.

## 3. Create the Notebook

1. kaggle.com -> **Notebooks -> New Notebook**.
2. Right panel **Settings**:
   - **Accelerator**: GPU (T4 x1) — single GPU.
   - **Internet**: ON (needed for the one-time ~98 MB ResNet50 weight download).
   - **Session timeout**: up to 12 hours.
3. **Add Input**: attach both `image-captioning-code` and `image-captioning-data`.
4. Open **Code**, paste the full contents of `kaggle/stage1_training.py` into one cell, run.

The cell sets the path env vars, copies the code into `/kaggle/working/imgcap`, installs the 3 lightweight deps, then runs:

```
python train.py --tag stage1 --epochs 10
```

## 4. Download results

When training ends, open the notebook's **Output** tab. The checkpoints are auto-saved there:

```
/kaggle/working/checkpoints/best.pt
/kaggle/working/checkpoints/last.pt
```

Download `best.pt` (and `last.pt` if you plan to resume). Copy them to your local `checkpoints/` dir for inference.

## Timing & resuming (12 h limit)

T4 x1, `num_workers=4`: the bottleneck is data loading (~4654 batches). Expect roughly 4–8 h for 10 epochs — usually fits one session.

If a session is interrupted:

1. Upload `last.pt` as a third dataset (e.g. `image-captioning-checkpoints`).
2. In the notebook cell, attach it and add `--resume`:
   ```python
   os.environ["IMG_CAP_DATA_ROOT"] = str(DATA_IN / "data")
   os.environ["IMG_CAP_CHECKPOINTS_DIR"] = str(WORK / "checkpoints")
   subprocess.run(["python", "train.py", "--tag", "stage1", "--epochs", "10",
                   "--resume", "/kaggle/input/image-captioning-checkpoints/last.pt"],
                  cwd=CODE_DIR, check=True)
   ```
   Training continues from the saved epoch.

## Notes

- `torch`/`torchvision` are **preinstalled** on Kaggle and are *not* reinstalled (`requirements-kaggle.txt` omits them). Reinstalling could break the GPU stack.
- No split/vocab recomputation happens on Kaggle: `splits/` and `captions.txt` ship inside the data zip, and `train.py` builds the same deterministic vocab locally.
- Offline weights fallback (only if a session ever has internet disabled): place `resnet50-0676ba61.pth` in the data zip under `torch/hub/checkpoints/` and add `os.environ["TORCH_HOME"] = str(DATA_IN / "torch")` in the cell.