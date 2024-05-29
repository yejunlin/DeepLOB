# Implement Of DeepLOB And It's Application Using C++
## Summary
This project is used to implement the DeepLOB refer to "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", published in IEEE Transactions on Singal Processing. It also contains another app used to generate some random value of the stock and use the model to predict the movement of the stock.
## Project Structure
### data_process.py
process the FI-2012 dataset, please unzip the file in data first
### model.py
Implement the DeepLOB model based on CNN-LSTM and a modified version AE-DeepLOB
### save_model_script.py
Transfer pytorch model to torch script
### test.py
Test the model
### train.py
Train the model
### main.cpp
Generate some data and use the model to predict the movement
### data
The FI-2012 data, please unzip the file before using
### CMakeLists.txt
cmake profile
### model.pt
an example of torch script using by main.cpp
### log
an example of output from main.cpp
### best_val_model
The trained torch model
## Notice
AE-DeepLOB is a modified version of DeepLOB, I introduced Auto Encoder to the vanilla model, but it has not been fully fine-tuned and tested due to limit CPU/GPU resources

Some code include the usage of CUDA, please uncomment the relate code if you are using CPU

The main.cpp needs a libtorch, please download the coordinate version from PyTorch. The version of libtorch and torch should be the same

The project runs on torch version 1.9.0+cu111, other version of torch may not be tested
