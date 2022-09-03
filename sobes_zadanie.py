#!/usr/bin/env python
# coding: utf-8

# In[601]:


#Импорт необходимых библиотек
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string

from gensim.utils import tokenize
from natasha import (Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)
from sklearn.metrics import pairwise_distances
from yargy import Parser, rule
from yargy.predicates import gram, dictionary

#Загрузка русских стоп-слов
stopwords_ru = stopwords.words("russian")

#Инициализация методов пакета Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

#Загрузка данных
dataframe=pd.read_csv('test_data.csv')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df=dataframe.copy()

def tokenize_text(raw_text: str):
    "Функция для токенизации текста"
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if ( i not in string.punctuation )]
    filtered_tokens = [i for i in tokens if ( i not in stopwords_ru )]
    return filtered_tokens

tokenized_text= df.text.apply(tokenize_text)
df = df.assign(tokenized=tokenized_text)

def get_manager_name(text):
    "Функция для получения имени менеджера"
    extractor = NamesExtractor(morph_vocab)
    matches = extractor(text.title())
    for match in matches:
        return(match.fact.first)

def get_company_name(text):
    "Функция для получения названия компании"
    company = rule (dictionary({'компания'}),gram('NOUN').repeatable())
    parser = Parser(company)
    for match in parser.findall(text):
        return([x.value for x in match.tokens])

#Распаковка диалогов из датасета
dialog=[]
for dlg in np.unique(df.dlg_id):
    dial=''
    for i in range(len(df[df['dlg_id']==dlg])):
        dial+=df[df['dlg_id']==dlg].iloc[i]['role']+': - '+df[df['dlg_id']==dlg].iloc[i]['text']+' \n '
    dialog.append(dial)

#Векторизация текста
vectorizer = CountVectorizer(tokenizer=tokenize_text)
# Шаблоны для векторного сравнения (не хватило времени придумать более изощренный способ. "Это Анастасия" - это костыль.)
greeting_to_compare='Здравствуйте, Добрый день'
farewall_to_compare='До свидания, всего доброго, всего хорошего'
introducing1_to_compare='Меня зовут'
introducing2_to_compare='Это анастасия'
df_text_values=np.append(df.text.values,[greeting_to_compare,farewall_to_compare,introducing1_to_compare,introducing2_to_compare])
document_matrix = vectorizer.fit_transform(df_text_values)

#Расчет косинусных расстояний
text_distance = 1-pairwise_distances(document_matrix, metric="cosine")

#Ранжирование по частоте индексов реплик
greeting_sorted_similarity = np.argsort(-text_distance[480,:])
farewall_sorted_similarity = np.argsort(-text_distance[481,:])
introducing1_sorted_similarity = np.argsort(-text_distance[482,:])
introducing2_sorted_similarity = np.argsort(-text_distance[483,:])

#Получение индексов
greeting_idx=list(greeting_sorted_similarity[1:12])
farewall_idx=list(farewall_sorted_similarity[1:8])
introducing_idx=list(introducing1_sorted_similarity[1:5])+list(introducing2_sorted_similarity[1:2])

#Создание колонок с флагами
df['greeting']=0
df['farewall']=0
df['introducing']=0
df.loc[greeting_idx,'greeting']=1
df.loc[farewall_idx,'farewall']=1
df.loc[introducing_idx,'introducing']=1

#Получение результата парсинга диалогов в виде словарей result_dlg_i
for dlg in np.unique(df.dlg_id):
    globals()['result_dlg_'+str(dlg)]={}
    globals()['result_dlg_'+str(dlg)]['greeting_speech']=list(df[(df.dlg_id==dlg)&(df.role=='manager')&(df.greeting==1)].text.values)
    globals()['result_dlg_'+str(dlg)]['introducing_speech']=list(df[(df.dlg_id==dlg)&(df.role=='manager')&(df.introducing==1)].text.values)
    
    globals()['result_dlg_'+str(dlg)]['manager_name']=get_manager_name(str(df[(df.dlg_id==dlg)&(df.role=='manager')&(df.introducing==1)].text.values))
    globals()['result_dlg_'+str(dlg)]['company_name']=get_company_name(str(df[(df.dlg_id==dlg)&(df.role=='manager')&(df.introducing==1)].text.values))
    globals()['result_dlg_'+str(dlg)]['farewall_speech']=list(df[(df.dlg_id==dlg)&(df.role=='manager')&(df.farewall==1)].text.values)
    if ((df[(df.dlg_id==1)&(df.role=='manager')].greeting.sum()>0)&(df[(df.dlg_id==1)&(df.role=='manager')].farewall.sum()>0)):
        globals()['result_dlg_'+str(dlg)]['check']='Менеджер поздоровался и попрощался'
    elif ((df[(df.dlg_id==1)&(df.role=='manager')].greeting.sum()==0)&(df[(df.dlg_id==1)&(df.role=='manager')].farewall.sum()>0)):
        globals()['result_dlg_'+str(dlg)]['check']='Менеджер поздоровался, но не попрощался'
    elif ((df[(df.dlg_id==1)&(df.role=='manager')].greeting.sum()>0)&(df[(df.dlg_id==1)&(df.role=='manager')].farewall.sum()==0)):
        globals()['result_dlg_'+str(dlg)]['check']='Менеджер попрощался, но не поздоровался'     
    elif ((df[(df.dlg_id==1)&(df.role=='manager')].greeting.sum()==0)&(df[(df.dlg_id==1)&(df.role=='manager')].farewall.sum()==0)):
        globals()['result_dlg_'+str(dlg)]['check']='Менеджер не поздоровался и не попрощался (бука)'

result_dlg_1

