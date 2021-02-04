# transfer-learning-CNN


I have added pytorch source code for two classifier models enabling transfer learning on custom image dataset.

- vgg16
- resnet50

Put the custom dataset on the same directory and indicate the path by passing the "--dir" argument.

The model can be selected as an argumet to the program by selecting the  "--model" argument.


How to run:

To train the model:

python3 transfer-cnn.py --model <model_name> --dir <path_to_dataset>

To view the classifier in action:

python3 image-inference.py
