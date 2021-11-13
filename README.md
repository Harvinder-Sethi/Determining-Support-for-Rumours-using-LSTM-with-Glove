# Determining-Support-for-Rumours-using-LSTM-with-Glove

## Problem Definition and Review
Social media has gradually evolved over the last 15 years to become a primary source of news. Social media, on the other hand, has become fertile ground for rumours, spreading them in a matter of minutes. An assertion that could be genuine or incorrect is classified as a rumour. False rumours can have a significant impact on a society's social, economic, and political stability, necessitating the development of tools to assist people, particularly journalists, in analysing the propagation of rumours and their impact on society, as well as determining their truth. Because Twitter and Reddit are two well-known social media platforms capable of disseminating breaking news, most rumour-related research employs the Twitter and reddit feed as a starting point, hence giving us the scope and space to build solutions for this problem and that is what I will be designing. 
I have done total of 4 literature reviews [1,2,3,4] for this task including the one in canvas. These all are directly based on determining rumour and veracity on mainly twitter, reddit and some on the other dataset as well. They have practiced different evaluation measures, feature selection technique and text pre-processing steps accordingly for support and veracity classification and the dataset in their papers. I read and gained some insights and added my knowledge to these and designed my algorithm as we have less train data and so said for the limitation of this, we can start to build a simple model that can be scaled in future.

## EDA & Evaluation Framework
Data Analysis- We have two datasets and I have combined twitter and reddit dataset, so we have more data to build our model. Checking the distribution of the data, we can see classes are imbalance and ideally, we will take f1 score (balance between precision & recall) into consideration [1] [1.2]. Checking the unique value counts of the class we noticed we have -1 for the source tweet, as we have less than 5% of those entries and to focus more on the reply, I decided to drop it. (We can have different approaches by appending the data of source Infront of all respective replies in a new column [3]). I have used regex, natural language toolkit nltk and python string package to do some cleaning (removing links, tokenizing the text, convert to lowercase, remove punctuations, remove all non-alphabetic characters including @, #) [2][3][4][12] 

Evaluation Metric- we can see classes slightly imbalance and ideally will take f1 score (balance between precision & recall) into consideration [1][2][3][4]. I have calculated F1 score from keras using backend [6]. 
Optimizer- It can be said that the choice of optimiser can dramatically change the performance and time of the model. One argument about the optimizers can be easily seen that SGD better generalizes than ADAM but ADAM converges faster. We tried Adam optimizer in base model and with InverseTimeDecay learning rate in hyperparameter tuning and is mostly considered in classification tasks, and according to our problem of classification and less time, data and resource we have selected ADAM.
Cost Function/Loss- Selecting a reasonable cost function according to problem is very important and its usually differential in deep learning problems. As our problem is not binary classification rather it is categorical as said being multi- class (more than 2). We have selected CategoricalCrossentropy as loss.

## Approach & Diagnostic Instrumentation
As this problem of determining support for rumours is not novel and has many solutions on the web, I did extensively investigate research papers [1][2][3][4]. Based on the findings and understanding as this is a many-one text classification task with less than usual data, we come up with a LSTM as baseline model [1][2][4] and then tune it accordingly to our desired result. Considering the data, resources, and literature reviews, for our problem we are expecting a base Accuracy score in between 75-80%.
As you can see in Fig1.1 we first started with setting notebook & packages, in this we imported all the required libraries, packages, setting up tensor board and our function to plot the learning curves. Further we did Data loading & EDA as explained above in Data Analysis, in that we manually encoded classes to change them to unique numerical values for further evaluation. Then Used keras tokenizer to generate the vocabulary for the data, created a list of texts, and used padded_batch method to zero pad the sequences to the length of the longest string in the batch (batch level, not dataset level). Then applied on tokenizer object to create our embedding matrix. 
Experiments, Tuning & Analysis
We then moved to Creating our Baseline model LSTM, according to literature reviews [1][2][3][4] and our knowledge from the course, we know that RNN are not good at capturing long range dependencies, and leads to vanishing gradient and lesser learning model while LSTM helps to capture long range dependencies and copes well with gradients, it can also be easily scaled to further larger classification (just little hyperparameter tuning and more data required), hence we selected LSTM Model. Coming to the model architecture, we used a 100-dimensional vector embedding matrix, purely based on the size of data and quantity of words we might have in that. We have used trainable =True in our embedding layer and we are learning the weights for the word vector matrix. In the lstm layer we have used units=32, as it reflects the dimensionality of the output space. Then we have a Dense output layer with 4 classes and SoftMax activation (multi-class classification), because it allows us to interpret the outputs as probabilities []. We then compiled the model for 25 Epochs with batch size of 32 (we kept these fixed throughout to compare results), we got an average F1 Score =0.81. We did not observed overfitting or underfitting as such, but we tried with decaying learning rate and dropout of 20% as well and we observed more smoother curves and we achieved roughly 0.78 F1 Score. 

We then moved to Transfer Learning with Glove Vocab, In English language the smallest unit with meaning is basically the words and hence word Level Encoding is more intuitive and preferred over character Level Encoding. While subword8k, 32k or other word encoding are decided basis on the dataset, simplicity and occurrences/need of words. We used 100-dimensional glove word embedding matrix trained on Wikipedia. We then read the glove vectors one by one from file then creating and storing it in a glove vector dictionary. Now I have transferred these glove word embedding vectors to our embedding matrix by mapping our vocabulary to glove's. We then updated the embedding layer with these weights and set trainable=False (we are now using glove weights). The rest of the model architecture is kept same as base model, we now notice the loss tends to decrease roughly by 15% and graphs are smoother [Fig.1.3], while we achieved an F1 Score of 0.78 [Fig.1.4]. we did not observed overfitting or underfitting as such, but we again tried this model with decaying learning rate and dropout of 20% as well (to compare for future extension of this model wrt. Baseline model) and we noticed after 20 epochs the loss tends to be increase a little than before but more stagnant giving a smooth loss graph [Fig.1.5] and we achieved roughly the same F1 Score= 0.78 but smoother F1 Graph [Fig.1.6].
Note- Since this is more like a Sentiment analysis many to one task with less data than usual, we don’t need to increase the complexity of model by stacking RNN cells on top of each other, and hence we used return sequences = False (default) across all above models.

## Ultimate Judgment & Limitations
 As we are close to our initial goal/expectation of performance of our model [1][2] we can stop here and finalise this as the best model we have, namely model_glove. We have then done our final prediction using this model on unseen test data split and got a similar F1 Score of 0.78 (approx.) [Fig.1.7]. On the contrary, we have tried different LSTM unit values [Fig.1.7] and noticed that decreasing the units more decreases the F1 score and accuracy and maybe reduces the model’s complexity more. Secondly Any of the baseline or glove model can be used for this specific task. While our baseline model was working fine in this case and was giving little high F1 and Accuracy, but we used model_glove as firstly not much difference, but with less loss and smoother graph and more importantly as we have glove vector embedding it can easily be further extended to cover more long-term dependencies via increasing the data and changing some hyper parameters. Meanwhile both these models are limited to only reply tweets and text classification, as we have dropped source. For a much larger dataset and complex model, we can maybe build more better model that takes two inputs i.e.. source text and corresponding reply to it as second input and then classify among the 4 classes (in this way we might store more long range dependencies by storing the context(source) of the replies).

Another drawback of detection of rumour in twitter and other datasets is that initial rumour is detected as a kind of false information propagation and then the task of rumour classification is performed. While multi-step classification in a supervised manner can solve this problem giving much better and true results. [4]

## References:
1.	Gorrell, G., Kochkina, E., Liakata, M., Aker, A., Zu- biaga, A., Bontcheva, K. and Derczynski, L., 2019, June. SemEval-2019 task 7: Ru- mourEval, determining rumour veracity and support forrumours. In Proceedings of the 13th International Workshop on Semantic Evaluation”
2.	Aclanthology.org. 2021. [online] Available at: <https://aclanthology.org/S17-2082.pdf> [Accessed 25 October 2021]. 
3.	Arxiv.org. 2021. [online] Available at: <https://arxiv.org/pdf/1611.06314.pdf> [Accessed 25 October 2021].
4.	Pathak, A., Mahajan, A., Singh, K., Patil, A. and Nair, A., 2021. Analysis of Techniques for Rumor Detection in Social Media.
5.	Analytics Vidhya. 2021. Twitter Sentiment Analysis | Implement Twitter Sentiment Analysis Model. [online] Available at: <https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/> [Accessed 25 October 2021].
6.	How to get accuracy, f. and Jayaraman, A., 2021. How to get accuracy, F1, precision and recall, for a keras model?. [online] Data Science Stack Exchange. Available at: <https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model> [Accessed 25 October 2021].
7.	Brownlee, J., 2021. How to Diagnose Overfitting and Underfitting of LSTM Models. [online] Machine Learning Mastery. Available at: <https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/> [Accessed 25 October 2021].
8.	dataset, P., 2021. Preventing overfitting of LSTM on small dataset. [online] Cross Validated. Available at: <https://stats.stackexchange.com/questions/204745/preventing-overfitting-of-lstm-on-small-dataset> [Accessed 25 October 2021].
9.	Medium. 2021. Choosing the right Hyperparameters for a simple LSTM using Keras. [online] Available at: <https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046?source=post_internal_links---------5----------------------------> [Accessed 25 October 2021].
10.	use?, H., 2021. How many LSTM cells should I use?. [online] Data Science Stack Exchange. Available at: <https://datascience.stackexchange.com/questions/16350/how-many-lstm-cells-should-i-use> [Accessed 25 October 2021].
11.	In Keras, w., Andersen, A. and Andersen, A., 2021. In Keras, what exactly am I configuring when I create a stateful `LSTM` layer with N `units`?. [online] Stack Overflow. Available at: <https://stackoverflow.com/questions/44273249/in-keras-what-exactly-am-i-configuring-when-i-create-a-stateful-lstm-layer-wi> [Accessed 25 October 2021].

## Appendix:

  
Fig.1.1						Fig.1.2	
![image](https://user-images.githubusercontent.com/29870980/141605188-8e2127d0-d7f0-4d6e-8bb2-979d836c232f.png) ![image](https://user-images.githubusercontent.com/29870980/141605195-7d407ab5-011f-42ad-85c7-d26b1cec76e4.png)


Fig.1.3						Fig.1.4

![image](https://user-images.githubusercontent.com/29870980/141605203-ff7d7828-f6c1-4212-a787-3f8093f6c005.png) ![image](https://user-images.githubusercontent.com/29870980/141605206-86d22251-5f35-406f-a0a1-ac866b0dfd40.png)

Fig.1.5
 
![image](https://user-images.githubusercontent.com/29870980/141605213-aa243118-fee2-4d51-8d4c-2b5782938374.png)

 
Fig.1.6

![image](https://user-images.githubusercontent.com/29870980/141605218-dc0ceff6-1657-4ea9-998e-89e1e87eaa0d.png)


Fig.1.7

![image](https://user-images.githubusercontent.com/29870980/141605222-b8a30c26-551f-462f-8319-ba89386840e2.png)


