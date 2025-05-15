- git clone
- install uv


```bash
uv venv --seed
source .venv/bin/activate
uv sync --no-build-isolation
uv run mim install mmcv

uv run mmdetection/tools/misc/download_dataset.py --dataset-name coco2017 --save-dir ./data/coco --unzip --delete
uv run mmdetection/tools/misc/download_dataset.py --dataset-name voc2012 --save-dir ./data --unzip --delete
uv run mmdetection/tools/misc/download_dataset.py --dataset-name voc2017 --save-dir ./data --unzip --delete
```


```bash
uv run src/detecbench/detecbench.py task:pgd
```

```bash
uv venv
```

```bash
uv venv
```
