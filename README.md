# Demo

This demo show you how to use libtorch, please reference https://pytorch.org/tutorials/advanced/cpp_export.html

## Requirements

- PyTorch (1.2.0)

- Libtorch

- OpenCV

  

## Build

```
python model.py

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/dell/libtorch ..
make
cd ..
```

## Test

![](./images/dog.jpg)

```
$ build/classifier resnet50.pt label.txt images/dog.jpg 
Class Result:
  beagle: 99.1227%
  Walker hound, Walker foxhound: 0.469355%
  English foxhound: 0.110916%

```
![](./images/shark.jpg)
```
$ build/classifier resnet50.pt label.txt images/shark.jpg 
Class Result:
  tiger shark, Galeocerdo cuvieri: 92.2599%
  great white shark, white shark, man-eater, man-eating shark: 5.9425%
  hammerhead, hammerhead shark: 1.77417%
  
```
