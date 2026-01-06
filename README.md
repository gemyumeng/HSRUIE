Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement
====
- Set up a conda environment
```
conda env create -f environment.yml
conda activate hsruie
```

- Test:
```
test datasets path: ./datasets/maps/testA
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
