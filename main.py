import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import collections
import spacy
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from DataCleaning import clean_data
# from DataPreprocessing import preprocess_data

# Change working directory
os.chdir(r"C:\Users\User\Documents\GitHub\UBD\SS4290 Project")
# os.chdir('/Users/faiz/Desktop/UBD Final Year/UBD/SS4290 Project')

# # DATA CLEANING
# df = clean_data(r"C:\Users\User\Documents\GitHub\UBD\SS4290 Project\project_data\complaints-2022-09-09_07_49.csv")
# # df = clean_data("/Users/faiz/Desktop/UBD Final Year/UBD/SS4290 Project/project_data/complaints-2022-09-09_07_49.csv")

# # DATA PREPROCESSING
# df = preprocess_data(df)
# df.to_csv('project_data/complaints_processed.csv')

# # FEATURE EXTRACTION (Bag of Words)
# from sklearn.feature_extraction.text import CountVectorizer


def get_data(n=10000):
    n_per_cat = int(n / 5)

    credit_card_df = df.loc[df['category'] == 'credit_card'].head(n_per_cat)
    retail_banking_df = df.loc[df['category']
                               == 'retail_banking'].head(n_per_cat)
    credit_reporting_df = df.loc[df['category']
                                 == 'credit_reporting'].head(n_per_cat)
    mortgages_and_loans_df = df.loc[df['category']
                                    == 'mortgages_and_loans'].head(n_per_cat)
    debt_collection_df = df.loc[df['category']
                                == 'debt_collection'].head(n_per_cat)

    # df = df.loc[:5000]

    new_df = pd.concat([credit_card_df, retail_banking_df,
                       credit_reporting_df, mortgages_and_loans_df, debt_collection_df])

    return new_df


# Train model (5000 records)
df = pd.read_csv('project_data/complaints_processed.csv')
df = get_data()
df = df.dropna()

# # Create a dataframe for each category to count their bag of words
categories = df['category'].unique()
# categories_bow_dict = dict.fromkeys(categories)
# categories_word_freq = dict.fromkeys(categories)

# for category in categories:
#     filtered_category = df[df['category'] == category]

#     CountVec = CountVectorizer(ngram_range=(1,1), min_df= 3,# to use bigrams ngram_range=(2,2)
#                                stop_words='english')
#     #transform
#     Count_data = CountVec.fit_transform(filtered_category['complaint'])

#     #create dataframe
#     cv_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())
#     categories_bow_dict[category] = cv_dataframe

#     # Word frequency
#     word_list = CountVec.get_feature_names_out()
#     count_list = np.asarray(Count_data.sum(axis=0))[0]
#     word_freq  = dict(zip(word_list, count_list))
#     categories_word_freq[category] = word_freq

# # FEATURE EXTRACTION (TF-IDF)

# categories_tfidf_dict = dict.fromkeys(categories)
# categories_tfidf_top = dict.fromkeys(categories)

# for category in categories:
#     filtered_category = df[df['category'] == category]

#     TfidfVec = TfidfVectorizer()
#     Tfidf_data = TfidfVec.fit_transform(filtered_category['complaint'])

#     tfidf_df = pd.DataFrame(data = Tfidf_data.toarray(), columns=TfidfVec.get_feature_names_out())
#     categories_tfidf_dict[category] = tfidf_df


# Change complaint cateogry to integers
category_dict = {'credit_reporting': 0, 'debt_collection': 1, 'mortgages_and_loans': 2,
                 'credit_card': 3, 'retail_banking': 4}
df['category'].replace(category_dict, inplace=True)
df = df.rename(columns={"category": "complaint_category",
               "complaint": "complaint_narrative"})

# Get length analysis for each complaints
# df['word_count'] = df["complaint_narrative"].apply(lambda x: len(str(x).split(" ")))
# df['char_count'] = df["complaint_narrative"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
# df['avg_word_length'] = df['char_count'] / df['word_count']

# Get sentiment of each complaints
# from textblob import TextBlob
# df["sentiment"] = df['complaint_narrative'].apply(lambda x:TextBlob(x).sentiment.polarity)

# Get NER of each complaints
# ner = spacy.load("en_core_web_lg")
# df["tags"] = df["complaint_narrative"].apply(
#     lambda x: [(tag.text, tag.label_) for tag in ner(x).ents])

# def utils_lst_count(lst):
#     dic_counter = collections.Counter()
#     for x in lst:
#         dic_counter[x] += 1
#     dic_counter = collections.OrderedDict(
#         sorted(dic_counter.items(),
#                 key=lambda x: x[1], reverse=True))
#     lst_count = [{key: value} for key, value in dic_counter.items()]
#     return lst_count

# # Count tags
# df["tags"] = df["tags"].apply(lambda x: utils_lst_count(x))

# def utils_ner_features(lst_dics_tuples, tag):
#     if len(lst_dics_tuples) > 0:
#         tag_type = []
#         for dic_tuples in lst_dics_tuples:
#             for tuple in dic_tuples:
#                 type, n = tuple[1], dic_tuples[tuple]
#                 tag_type = tag_type + [type]*n
#                 dic_counter = collections.Counter()
#                 for x in tag_type:
#                     dic_counter[x] += 1
#         return dic_counter[tag]
#     else:
#         return 0

# # Extract features
# tags_set = []
# for lst in df["tags"].tolist():
#     for dic in lst:
#         for k in dic.keys():
#             tags_set.append(k[1])
# tags_set = list(set(tags_set))
# for feature in tags_set:
#     df["tags_"+feature] = df["tags"].apply(lambda x:
#                                             utils_ner_features(x, feature))

# # df.to_csv('NER_tags_extracted.csv', encoding='utf-8', index=False)
# df = pd.read_csv('NER_tags_extracted copy.csv')
# df = df.drop(['tags_PERCENT', 'tags_EVENT', 'tags_LOC', 'tags_QUANTITY',
#              'tags_LAW', 'tags_FAC', 'tags_WORK_OF_ART', 'tags_LANGUAGE'], axis=1)

# Split data to train and test sets
X = df.drop(['complaint_category'], axis=1)
y = df['complaint_category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2127)

# Create word_count datafarme for each train and test set to be converted to sparse matrix
# wordcount_train = X_train.filter(['word_count','char_count', 'avg_word_length'], axis=1)
# wordcount_test =  X_test.filter(['word_count','char_count', 'avg_word_length'], axis=1)

# Create sentiment dataframe for each train and test set to be converted to sparse matrix
# sentiment_train = X_train.filter(['sentiment'], axis=1)
# sentiment_test =  X_test.filter(['sentiment'], axis=1)

# Create NER dataframe for each train and test set to be converted to sparse matrix
# NER_train = X_train.filter(regex='tags_', axis=1)
# NER_test = X_test.filter(regex='tags_', axis=1)


# Creating TF-IDF values of compalaints to be used for baseline NB and SVM models
baseline_vectorizer = TfidfVectorizer()
vect_X_train = baseline_vectorizer.fit_transform(
    X_train['complaint_narrative'])
vect_X_test = baseline_vectorizer.transform(X_test['complaint_narrative'])

# Convert wordcount df to a sparse matrix to be combined with TF-IDF
# wordcount_sparse_train = csr_matrix(wordcount_train.values)
# wordcount_sparse_test = csr_matrix(wordcount_test.values)

# Convert sentiment df to a sparse matrix to be combined with TF-IDF
# sentiment_sparse_train = csr_matrix(sentiment_train)
# sentiment_sparse_test = csr_matrix(sentiment_test)

# Convert TF-IDF and NER tags df to a sparse matrix to be combined with TF-IDF
# ner_sparse_train = csr_matrix(NER_train)
# ner_sparse_test = csr_matrix(NER_test)

# Combine TF-IDF with other features (word_count, sentiment, NER tags)
# combined_features_train = hstack([vect_X_train])
# combined_features_test = hstack([vect_X_test])

# Naive Bayes baseline model
# nb_baseline = MultinomialNB()
# nb_baseline.fit(vect_X_train, y_train)
# nb_baseline_pred = nb_baseline.predict(vect_X_test)

# print(
#     f'Naive Bayes (Baseline) using TF-IDF Classifcation Report:\n{classification_report(y_test, nb_baseline_pred)}')

# print(
#     f'Naive Bayes (Baseline) using TF-IDF Accuracy: {accuracy_score(y_test, nb_baseline_pred)}')

# # Get the confusion matrix
# nb_cfm = confusion_matrix(y_test, nb_baseline_pred)
# nb_cfm_plt = sns.heatmap(nb_cfm, annot=True, cmap="YlGnBu",  fmt='g')
# nb_cfm_plt = nb_cfm_plt.set(xlabel='Predicted Label', ylabel='True Label',
#                             title=f'Multinomial Naive Bayes (Baseline) using TF-IDF | Accuracy:{round(accuracy_score(y_test, nb_baseline_pred)*100, 2)}%')

# # SVM baseline model
# svm_baseline = SVC()
# svm_baseline.fit(vect_X_train, y_train)
# svm_baseline_pred = svm_baseline.predict(vect_X_test)

# print(
#     f'SVM (Baseline) using TF-IDF Classifcation Report:\n{classification_report(y_test, svm_baseline_pred)}')
# print(
#     f'SVM (Baseline) using TF-IDF Accuracy: {accuracy_score(y_test, svm_baseline_pred)}')

# # Get the confusion matrix
# svm_cfm = confusion_matrix(y_test, svm_baseline_pred)
# svm_cfm_plt = sns.heatmap(svm_cfm, annot=True, cmap="YlGnBu",  fmt='g')
# svm_cfm_plt = svm_cfm_plt.set(xlabel='Predicted Label', ylabel='True Label',
#                               title=f'SVM (Baseline) using TF-IDF | Accuracy:{round(accuracy_score(y_test, svm_baseline_pred)*100, 2)}%')

# Save model
# import pickle
# pickle.dump(svm_baseline, open('project_data/svm_model.pkl', 'wb'))
# pickle.dump(baseline_vectorizer, open('project_data/vectorizer.pkl', 'wb'))

# Find best parameter to pass into TF-IDF of each model Naive Bayes and SVM

# nb_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('nb', MultinomialNB())
# ])

# svm_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('svm', SVC())
# ])

# parameters = {
#     'tfidf__min_df': (1, 5, 10),
#     'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
#     'tfidf__max_features': (1000, 10000, 100000),
# }

# nb_parameters = {
#     'nb__alpha': (0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000),
# }

# svm_parameters = {
#     'svm__C': (0.1, 1, 10, 100, 1000),
#     'svm__gamma': (1, 0.1, 0.01, 0.001, 0.0001),
# }

# grid_search_nb = GridSearchCV(
#     nb_pipeline, param_grid = {**parameters, **nb_parameters}, cv=2, verbose=3, scoring='accuracy')
# grid_search_nb.fit(X_train['complaint_narrative'], y_train)

# print("Best parameters set (Naive Bayes):")
# print(grid_search_nb.best_params_)

# grid_search_svm = GridSearchCV(
#     svm_pipeline, param_grid = {**parameters, **svm_parameters}, cv=2, verbose=3, scoring='accuracy')
# grid_search_svm.fit(X_train['complaint_narrative'], y_train)

# print("Best parameters set (SVM):")
# print(grid_search_svm.best_params_)

# Fit best parameters for Naive Bayes found from GridSearchCV into TF-IDF
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=100000)
vect_X_train = vectorizer.fit_transform(X_train['complaint_narrative'])
vect_X_test = vectorizer.transform(X_test['complaint_narrative'])

# Train Naive Bayes model
nb_cls = MultinomialNB(alpha=0.1)
nb_cls.fit(vect_X_train, y_train)
nb_pred = nb_cls.predict(vect_X_test)

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score, average='weighted'),
           'recall' : make_scorer(recall_score, average='weighted'), 
           'f1_score' : make_scorer(f1_score, average='weighted')}

# Evaluate Naive Bayes model using Cross Validation using 5 K-Fold
scores = cross_validate(nb_cls, vect_X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=False)
print('-- Naive Bayes Performance --')
print('Training Accuracy :', scores['train_accuracy'])
print("Average Training Accuracy: {:.2f}".format(
    scores["train_accuracy"].mean()*100))
print('Validation Accuracy :', scores['test_accuracy'])
print("Average Validation Accuracy: {:.2f}".format(
    scores["test_f1_score"].mean()*100))

# Plot confusion metrics and classfication report for Naive Bayes

# Print classification report
print(classification_report(y_test, nb_pred))
print(f'Naive Bayes (Tuned) Accuracy: {accuracy_score(y_test, nb_pred)}')


# Get the confusion matrix
nb_cfm = confusion_matrix(y_test, nb_pred)
nb_cfm_plt = sns.heatmap(nb_cfm, annot=True, cmap="YlGnBu",  fmt='g')
nb_cfm_plt.set(xlabel='Predicted Label', ylabel='True Label',
               title='Multinomial Naive Bayes Confusion Matrix')

# Fit best parameters for SVM found from GridSearchCV into TF-IDF
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=100000)
vect_X_train = vectorizer.fit_transform(X_train['complaint_narrative'])
vect_X_test = vectorizer.transform(X_test['complaint_narrative'])

# Train SVM model
svm_cls = SVC(C=10, gamma=1, kernel='rbf')
svm_cls.fit(vect_X_train, y_train)
svm_pred = svm_cls.predict(vect_X_test)

# Evaluate SVM model using Cross Validation using 5 K-Fold
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score, average='weighted'),
           'recall' : make_scorer(recall_score, average='weighted'), 
           'f1_score' : make_scorer(f1_score, average='weighted')}

scores = cross_validate(svm_cls, vect_X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True, verbose=3)
print('-- SVM Performance --')
print('Training Accuracy :', scores['train_accuracy'])
print("Average Training Accuracy: {:.2f}".format(
    scores["train_accuracy"].mean()*100))
print('Validation Accuracy :', scores['test_accuracy'])
print("Average Validation Accuracy: {:.2f}".format(
    scores["test_recall"].mean()*100))

# Plot confusion metrics and classfication report for SVM
# Print classification report
print(classification_report(y_test, svm_pred))
print(f'SVM (Tuned) Accuracy: {accuracy_score(y_test, svm_pred)}')


# Get the confusion matrix
svm_cfm = confusion_matrix(y_test, svm_pred)
svm_cfm_plt = sns.heatmap(svm_cfm, annot=True, cmap="YlGnBu",  fmt='g')
svm_cfm_plt.set(xlabel='Predicted Label', ylabel='True Label',
                title='SVM Confusion Matrix')

# function to return key for any value


def get_key(val):
    for key, value in category_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

