# CNDesc (under review)

To do：
- [x] Evaluation code for CNDesc
- [x] Trained model 
- [ ] Training code (After the paper is accepted.)


# Requirement
```
pip install -r requirement.txt,
```

# Quick start
HPatches Image Matching Benchmark

1.Download the trained model: https://drive.google.com/file/d/16mVPNgYgmAgJ-DlmA7zC8lRffH4l0l5x/view?usp=sharing
and place it in the "ckpt/cndesc".


2.Download the HPatches dataset：

```
cd evaluation_hpatch/hpatches_sequences
bash download.sh
```
3.Extract local descriptors：
```
cd evaluation_hpatch
CUDA_VISIBLE_DEVICES=0 python export.py  --tag [Descriptor_suffix_name] --config ../configs/CNDesc_extract.yaml
```
4.Evaluation
```
cd evaluation_benchmark
python hpatch_benchmark.py --config ../configs/hpatches_benchmark.yaml
```
