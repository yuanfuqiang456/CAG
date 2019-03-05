# CAG
CAG means Chronic Atrophic Gastritis. This project use CNN to dectect CAG .

This research was done together by YuanFuqiang and the doctors Li Fengxia,Zhang Yaqiong of the Department of Gastroenterology, Shanxi Provincial People's Hospital.
The purpose is to detect the final chronic atrophic gastritis of the gastroscopic image.

This project uses deep learning techniques to complete this research.Specially,We have more than 2000 pictures and we use DenseNet161 to  reach the state-of-art in the dectecting of Gastric antrum image. The recognition accuracy of chronic atrophic gastritis is more than 98%,and level of sensitivity is 95.9%，the specificity
is 94%。
We have drawn the ROC curve and p-R curve of the model separately.
The POC curce is like below .

 ![image](https://github.com/yuanfuqiang456/CAG/blob/master/pic/ROC.png?raw=true)
 
 The p-R curve is also prefect.
 
 ![image](https://github.com/yuanfuqiang456/CAG/blob/master/pic/P-R.png?raw=true)
 
 In order to find the right model,We use different models for training。And finally ,we have drawn curves for different model accuracy rates.
 
  ![image](https://github.com/yuanfuqiang456/CAG/blob/master/pic/acc-lines.png?raw=true)
  
  In order to make an interpretability study on the model, the model learning results are visualized, and a model identification heat map is generated.
  
  ![image](https://github.com/yuanfuqiang456/CAG/blob/master/pic/visual.png?raw=true)
  
  #How to use this project.
   This project can be runned in the environment of pytorch_0.4 and tensorboard .Tensorboard is used to display parameter changes during program execution.
  The project architecture is showed as below.
```
├── checkpoints/ 
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── get_data.sh
├── models/
│   ├── __init__.py
│   ├── AlexNet.py
│   ├── BasicModule.py
│   └── ResNet34.py
│   └── densenet.py
└── utils/
│   ├── __init__.py
│   └── logger.py
└── logger/
├── config.py
├── main.py
|—— T_AlexNet.py
|—— T_Inception.py
├── T_VGG.py
├── T_ResNet.py
├── README.md
```
  The Program entrance is main.py or other programs start with T-.The T- means Test of different models .The floder checkpoints is used to save model after train,and data is leaved with different pictures wu used .The models is model we used and changed according to our problems.and the logger.py and floder utils is used by tensorboard to save paramaters change during training .
  If you want to visual you result of Gastric antrum image，you can change function “if __name__=='__main__':” of main.py or T_XXX.py.We have uploaded a trained model in this project of checkpoints floder.Yu can use it directly.So gopd luck!
  if you hace any question,you can concat with me.


 
