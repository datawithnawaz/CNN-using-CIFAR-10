# CIFAR-10 Image Classification with CNN (PyTorch)

I built a image classifier using Convolutional Neural Networks. The goal was to train a model on the CIFAR-10 dataset and get it to correctly identify what object is in a given image. Sounds simple but trust me there was a lot of trial and error involved.

## What is CIFAR-10?

CIFAR-10 is basically a collection of 60,000 small images (32x32 pixels) each belonging to one of 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

50,000 of these are used for training the model and the remaining 10,000 are used to test how well it actually learned.


## What's in this notebook?

The notebook is split into 3 main parts.

### Part 1 — Building and Training a Basic CNN

I built a straightforward CNN from scratch using PyTorch. Nothing fancy here, just 3 convolutional layers followed by 2 fully connected layers. The model trains on the CIFAR-10 training data and then gets evaluated on the test set.

You'll see:
- The training loss going down over 20 epochs (that's a good sign, means the model is actually learning something)
- Final test accuracy printed out at the end
- A breakdown of accuracy per class so you can see which ones the model struggles with (spoiler: cats and dogs tend to confuse it)

### Part 2 — Technical Write-up

This part explains the design decisions I made — why I picked the number of layers I did, what loss function I used and why, and how the accuracy changed throughout training. All written in plain english inside the notebook itself.

### Part 3 — Trying Out Advanced Stuff

This is the more research-heavy part. I experimented with two techniques that are commonly used in real-world CNN architectures:

**Residual Connections (Skip Connections)**
Instead of just stacking layers on top of each other, skip connections let the output of an earlier layer "skip" ahead and get added to a later layer. The idea is that if a layer doesn't have anything useful to add, the network can just pass the original signal through. In practice this makes training more stable, especially in deeper networks.

**Batch Normalization**
This basically normalizes the outputs of each layer before passing them to the next one. Think of it like making sure the numbers flowing through the network don't go all over the place and become too big or too small. It speeds up training noticeably and the loss curves become much smoother.

For both of these I compared them against the basic CNN to show whether they actually helped or not.


## How to Run This

You don't need to download any data manually. PyTorch will handle that automatically when you run the notebook. Google colab is recommended because this training needs some good spec PC to run otherwise it may crash.

**Requirements:**

```
torch
torchvision
matplotlib
numpy
```

Install them with:

```bash
pip install torch torchvision matplotlib numpy
```

Then just open the notebook and run all cells from top to bottom. It should work on any machine — I made sure of that. If you have a GPU it will use it automatically, if not it will just run on CPU (which is slower but still works fine).

> Heads up — training all three models for 20 epochs each can take anywhere from 15 to 30 minutes depending on your hardware. Grab a coffee.

---

## Files in This Repo

```
├── CNN-nawaz.ipynb   # main notebook with all the code and write-ups
├── CNN-nawaz.html    # html source for overview
├── README.md                      # this file
```

The notebook will also save some figures as PNG files when you run it:
- `basic_cnn_training.png`
- `residual_cnn_training.png`
- `bn_cnn_training.png`
- `comparison_basic_vs_residual.png`
- `comparison_basic_vs_bn.png`
- `all_models_comparison.png`

---

## A Few Things I Noticed While Building This

- The model learns really fast in the first 5 epochs and then slows down a lot — this is totally normal
- Cats and dogs are genuinely hard for the model to tell apart, which honestly makes sense given how similar they look in a 32x32 image
- Batch Normalization made a noticeable difference in how smooth the training was. Without it the loss curve was a bit all over the place in early epochs
- Residual connections didn't make a huge difference for a 3-layer network but the concept makes a lot more sense when you think about very deep networks with 50+ layers

---

## References

- He et al. (2016) — Deep Residual Learning for Image Recognition *(original ResNet paper)*
- Ioffe & Szegedy (2015) — Batch Normalization: Accelerating Deep Network Training
- PyTorch official docs — https://pytorch.org/docs/stable/index.html
- CIFAR-10 dataset — https://www.cs.toronto.edu/~kriz/cifar.html

---
