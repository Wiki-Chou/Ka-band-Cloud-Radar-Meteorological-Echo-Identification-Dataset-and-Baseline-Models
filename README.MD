# Ka-band Cloud Radar Meteorological Echo Identification Dataset and Baseline Models for Deep Learning Algorithms

This project is a deep learning project for meteorological radar echo recognition, implementing various Unet architecture models.

## Directory Structure

## File Descriptions

- `comparision.ipynb`: Jupyter Notebook for comparing fuzzy logic, BP neural network, and deep learning algorithms.
- `loss/`: Contains implementations of different loss functions.
  - `bceLoss.py`: Binary Cross-Entropy loss function.
  - `diceLoss.py`: Dice loss function.
  - `msssimLoss.py`: Multi-Scale Structural Similarity loss function.
- `model/`: Contains different trained models for reference.
  - `TransUnet/`: Transformer U-Net model.
  - `Unet/`: U-Net model.
  - `UnetSE/`: U-Net model with Squeeze-and-Excitation module.
  - `UnetSE_LDR_randomDROP/`: U-Net SE model with random dropout.
- `model_BP.joblib`: Trained model file.
- `output/`: Output results directory.
- `predict_visualization.py`: Script for prediction and visualization.
- `runs/`: Logs and results of training runs.
- `train.py`: Training script.
- `tran4BP.ipynb`: Jupyter Notebook for training BP neural network.
- `unet/`: Contains U-Net related code.
  - `init_weights.py`: Script for initializing weights.
  - `layers.py`: Script for defining network layers.
  - `SEUnet.py`: U-Net implementation with Squeeze-and-Excitation module.
  - `transunet.py`: TransUnet implementation.
  - `unet_model.py`: Classic Unet implementation.
- `utils/`: Contains utility functions and dataset processing code.

## Installation and Usage

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Train the model:
    ```sh
    python train.py
    ```

    Example:
    ```sh
    python train.py -g 0 -u se -e 50 -b 4 -l 0.00001 -f model.pth -s 1
    ```

    Argument descriptions:
    - `-g`, `--gpu_id`: GPU ID, default is 0.
    - `-u`, `--unet_type`: U-Net model type, options are `v1`, `v2`, `v3`, `trans`, `se`, default is `se`.
    - `-e`, `--epochs`: Number of epochs, default is 50.
    - `-b`, `--batch-size`: Batch size, default is 4.
    - `-l`, `--learning-rate`: Learning rate, default is 0.00001.
    - `-f`, `--load`: Load model from a .pth file, default is False.
    - `-s`, `--scale`: Downscaling factor of the images, default is 1.

4. Predict and visualize:
    ```sh
    python predict_visualization.py --input <input_image_path> --output <output_directory>
    ```

    Example:
    ```sh
    python predict_visualization.py --input ./data/test/imgs --output ./output
    ```

    Argument descriptions:
    - `--gpu_id`, `-g`: GPU ID, default is 0.
    - `--unet_type`, `-u`: U-Net model type, options are `v1`, `v2`, `v3`, `trans`, `se`, default is `se`.
    - `--input`, `-i`: Input image file paths, default is `./data/test/imgs`.
    - `--mask`, `-m`: Input mask file paths, default is `./data/test/masks`.
    - `--output`, `-o`: Output directories for results, default is `output`.
    - `--mask-threshold`, `-t`: Threshold for mask prediction, default is 0.5.
    - `--scale`, `-s`: Scale factor for input images, default is 1.

## Performance

### Accuracy Table

| Model        | mIOU   | Dice   | Precision | Recall | F1     | Time(s) | GPU Time(s) |
|--------------|--------|--------|-----------|--------|--------|---------|-------------|
| FL           | 0.7411 | 0.8211 | 0.8005    | 0.8667 | 0.8211 | 10.4643 |             |
| BP           | 0.7618 | 0.8322 | 0.9037    | 0.7982 | 0.8322 | 19.6090 |             |
| Unet         | 0.9668 | 0.9827 | 0.9793    | 0.9862 | 0.9827 | 5.3555  | 0.4815      |
| TransUnet    | 0.8734 | 0.9271 | 0.8970    | 0.9640 | 0.9271 | 11.2552 | 0.5297      |
| Unet+SE      | 0.9688 | 0.9838 | 0.9853    | 0.9822 | 0.9838 | 5.5242  | 0.5022      |

### Image Description

![Fig. 1](figure/fig1.png)

**Fig. 1** Comparison of clear-air echo identification capability under complex weather conditions: (a) simple case 1; (b) simple case 2; (c) drizzle; (d) rain; (e) low cloud.

## Connection

Weijie Zou 2016301610343@whu.edu.cn

## License

This project is licensed under the MIT License - see the [LICENSE](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22f%3A%5C%5CWorkspace%5C%5CProjects%5C%5C%E6%B0%94%E8%B1%A1%E5%B1%80%E6%8A%80%E8%83%BD%E5%A4%A7%E8%B5%9B%5C%5CKa%20Radar%20Meteorological%20Echo%20Identification%5C%5CLICENSE%22%2C%22_sep%22%3A1%2C%22path%22%3A%22%2FF%3A%2FWorkspace%2FProjects%2F%E6%B0%94%E8%B1%A1%E5%B1%80%E6%8A%80%E8%83%BD%E5%A4%A7%E8%B5%9B%2FKa%20Radar%20Meteorological%20Echo%20Identification%2FLICENSE%22%2C%22scheme%22%3A%22file%22%7D%7D) file for details.
