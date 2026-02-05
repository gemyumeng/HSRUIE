Task-Driven Underwater Image Enhancement via Hierarchical Semantic Refinement
====
- Set up a conda environment
```
conda env create -f environment.yml
conda activate hsruie
```

- checkpoints
```
download latest_net_NC.pth (./checkpoints/maps_cyclegan/) at: https://drive.google.com/file/d/1SgiK6fBHXd2yICBhNNfoncUWjSUxslYe/view?usp=sharing

download vgg19-dcbb9e9d.pth (./model_data/maps_cyclegan/) at: https://drive.google.com/file/d/1eRwpDSR8PlJKDuoG4JvBN9fOGfasaN51/view?usp=sharing

```

- Test:
```
test datasets path: ./datasets/maps/testA
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
