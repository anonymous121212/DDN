# Deep Distribution Network
* **Tensorflow 1.12.0**
* Python 3.6.9
* CUDA 9.0+ (For GPU)

# Instructions
To run the experiment with default parameters:
```
$ cd src
$ python main.py
```
You can change the all the parameters in `src/config.py`. To run on different datasets, for example, set `train_filename = "../data/ml-1m/train_users.dat"` to `train_filename = "../data/lastfm/train_users.dat"`. To test on the cold-start setting, for example, set `train_filename = "../data/ml-1m/train_users.dat"` to `train_filename = "../data/ml-1m/train_cold_users.dat"`. 

After running the experiments, you will see output like:
```
Epoch 0 loss 693.143555
Epoch 1 loss 693.110840
Epoch 2 loss 693.019043
Epoch 3 loss 692.881653
Epoch 4 loss 692.713257
Epoch 5 loss 692.430481
Epoch 6 loss 692.038086
Epoch 7 loss 691.646606
Epoch 8 loss 690.938965
Epoch 9 loss 689.866699
Epoch 10 loss 689.058960
Epoch 11 loss 687.606567
Epoch 12 loss 686.247437
Epoch 13 loss 683.669128
Epoch 14 loss 681.659180
Epoch 15 loss 678.146118
Epoch 16 loss 674.974182
Epoch 17 loss 669.581360
Epoch 18 loss 663.583801
Epoch 19 loss 657.597839
Epoch 20 loss 647.040649
Epoch 21 loss 641.907349
Epoch 22 loss 631.411621
Epoch 23 loss 619.181641
Epoch 24 loss 605.545410
Epoch 25 loss 588.443176
Epoch 26 loss 566.390869
Epoch 27 loss 551.436218
Epoch 28 loss 528.101990
Epoch 29 loss 512.296326
Epoch 30 loss 483.476990
Epoch 31 loss 457.129822
Epoch 32 loss 419.286102
```
