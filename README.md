# Solafune Solar Panel Detection

### Download and Install Miniconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh

### Install Nvidia drivers

First we can check our hardware, architecture, and distro with

```Bash
lspci | grep -i nvidia
uname -m && cat /etc/*release
```

which should show several NVIDIA devices, x86_64, and then Ubuntu 22.04 or some such.

Install build dependencies:

```Bash
sudo apt install gcc
sudo apt install linux-headers-$(uname -r)
```

On these systems we additionally had to remove an outdated signing key:
```Bash
sudo apt-key del 7fa2af80
```

Install the latest nvidia keyring:
```Bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
```
Finally, update sources, install CUDA, and reboot:
```Bash
sudo apt update
sudo apt install cuda
sudo reboot
```

Install environment:
Install Torch and Cuda
```Bash
pip install -r requirements.txt 
```
## Datasets

The repository has custom build model and dataloader that allows us more flexibility to merge and and more datasets in the future.

The Dataset is divided in a Train/Valid/Test CSV file that contains the path of the input target (RGB), targets (sengmentation mask and Depth estimation) and Dataset Name.

The CSV file is organized as follow:

![alt text](assets/dataset_info.png)

The CSV file are stored in: **data_splits** folder:

- test_path.csv
- train_path.csv
- valid_path.csv

In order to generate those files please refer to the Jupyter Nobook called **multitask_learning_dataset.ipynb**

During the training or evaluation phase the associated csv file is called:


```python
# Load train and test data path
train_path = pd.read_csv('data_splits/train_path.csv')
valid_path = pd.read_csv('data_splits/test_path.csv')
print("Number of Training data {0:d}".format(len(train_path)))
print("------")
print("Number of Validation data {0:d}".format(len(valid_path)))
print("------")
train_dataset = TrainDataset( df_path= train_path, transforms= None)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
eval_dataset = EvalDataset(df_path= valid_path, transforms=None)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=8, shuffle=False)
```

## Train model

```Bash
python train.py
```
Training Losss and Validation Loss can be access via a Tensorboard: http://10.0.0.6:8888/?darkMode=true#images


<!-- ![alt text](assets/tensorboard.png) -->

During the training few folders are created:
- **training_prediction**: Create a folder with the starting date of the training with the following data:
    - Image of RGB | target mask | predicted mask | target depth | predicted depth\

    <!-- ![alt text](assets/training_prediction.png) -->


    - validation metrics log html format

    <!-- ![alt text](assets/validation_metrics.png) -->
    - when the training is done all the validation metrics are saved in a csv file

- **result**: Create a folder with the starting date of the training and store the checkpoint at for each epoch. At the end of the training the best checkpoint will be saved as **best.pth


## Eval model

```Bash
python eval.py -i [input] -c [checkpoint]
```

where:

- **input**: csv file which contain path test data
- **checkpoint**: Model path.
___

During the evaluation inference, a folder **evaluation** is created. For each evaluation, a folder with the date of the inference is created. In that folder all inference output is saved.

<!-- ![alt text](assets/inference_output.png) -->
___

The folder also contain all the evaluation metrics:

```json



```

