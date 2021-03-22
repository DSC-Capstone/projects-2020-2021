# DSC180B-NER-Project
This project focuses on the task of document classification using a BBC News Dataset and a 20 News Group Dataset. We implemented various feature based classification models and compared the results. We have analysed the advantages and shortcomings of each method.

## Webpage
* https://dsc180b-a04-capstone-group-06.github.io/News-Classification-Webpage/

## Datasets Used
* BBC news: https://www.kaggle.com/pariza/bbc-news-summary </br>
  * Download this dataset
* 20 news group: http://qwone.com/~jason/20Newsgroups/ 
  * This dataset is fetched by using the sklearn package
## Environment Required
* Please use the docker image: ``` littlestone111/dsc180b-ner-project  ```

## Run
```
$ launch-180.sh -i littlestone111/dsc180b-ner-project -G [group]
$ python run.py [all] [preprocessing] [autophrase] [model] [test]
```

```test``` :        target will build the Tf-Idf models on the small subset of 20 new groups dataset and save the models to the model folder.</br>
```all```:          target will run everthing inlcuded in project, and return the final prediction on the test dataset for document classification.</br>
```preprocessing```: target will preprocess 20 news group data for AutoPhrase, so that they can be used for training the model.</br>
```autophrase```:   target will run Professor Shang's Autophrase model to extract quality phrases from the dataset.</br>
```model```:        target will build the SVM+ALL+TF-IDF combined vocab list model for 20 news group dataset. </br>

</br>
Output: <br>

* ```model.pkl```: the parameter of the final model.

## Group Members
* Rachel Ung
* Siyu Dong
* Yang Li

## Our Findings

The BERT classification on the five-class BBC News dataset does not outperform any of our implemented models. From our results table, we observed that our models have F1-Score and Accuracy performances at around 0.95, indicating they are high-performing classifiers. The best of them is the SVM+ALL(TF-IDF) classifier, or the Support Vector Machine with the All Vector Vocabulary List and Tf-Idf Representations, which uses the vocabulary from both NER results and AutoPhrase results. Because the quality phrases between different domains are likely to differ, we expect these results to be optimal features for our predictors. 

For the 20 News Group dataset, the SVM+ALL(TF-IDF) classifer also outperformed the other models, with the F1-Score and Accuracy being 0.84. Considering the classes are huge (i.e. 20 classes), these results verify our model is high-performing. Applying our best model on the five-class BBC News dataset, we attained a F1-Score at 0.9525, and Accuracy at 0.9528; while for the 20 News Group, we yielded a F1-Score at 0.8463 and Accuracy at 0.8478. 




