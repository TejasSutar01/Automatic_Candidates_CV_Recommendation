from django.shortcuts import render
import os
import pandas as pd

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
#Project packages
import numpy as np
import glob
import PyPDF2
from regex import B
import textract
import re
import string
import pandas as pd
import pdfplumber
import re
import os
import docx
import re
import nltk
from CV_Screen_app.serializers import CV_ScreenSerializer
import math
import contractions
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from collections import Counter
from nltk.tokenize import word_tokenize
import gensim
import docx2txt
from gensim.models.phrases import Phraser, Phrases
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def index(request):
    context={'A':"Hello World"}
    return render(request,'index.html',context)
    # return HttpResponse("hello World")
 
def predictCV(request):
    print(request)
    if request.method== "POST":
        temp={}
        temp=request.POST.get('skills')

        print("$$$$$$$$",temp)
        print("$$$$$$$$",type(temp))

        dirname = "/Volumes/DATA/CV_Screen/Django_API_HTML/CV_Screen_Project/All_CVs"  
        dfs=pd.DataFrame()
        file_paths = []
        lst=[]
        my_list=[]
        similarity_percentage=[]
        def getList(dict):
            return list(dict.keys()) 
        
        for item in os.listdir(dirname):
                print(":::::::::::::::::",item)
                if not item.startswith('.'):
                    file_paths.append(os.path.abspath(os.path.join(dirname, item)))
                    # print("------------",file_paths)
                    aa=item[:-4]
                    lst.append(aa)
        var=temp
        skills = ''.join(map(str, var))
        dfs['All_paths']=file_paths
        dfs['Names']=lst
        dfs['Skills']=dfs.apply(lambda x: skills, axis=1)
        print("##########Skills##############",dfs)

        for i in dfs['All_paths']:
            if '.pdf' in i:
#         print('\\\\\----If-----\\\\\\',i)
                pdf = pdfplumber.open(i)
                page = pdf.pages[0]
                text = page.extract_text()
                mystring = text.replace('\n', ' ').replace('\r', '')
                my_list.append(mystring)
            elif '.docx' in i:
#         print('\\\\\----Else-----\\\\\\',i)
        
                my_text = docx2txt.process(i)
                newtext = my_text.replace('\n', ' ').replace('\r', '')
                # print('=====Docs===',newtext)
                my_list.append(newtext)
            else:
                print("File not found")

        dfs['Extracted_text']=my_list

        def preprocess(sentence):

            sentence=str(sentence)
            sentence = sentence.lower()
            sentence = contractions.fix(sentence)
            sentence = sentence.replace('{html}',"") 
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', sentence)
            rem_email = re.sub('\S+@+\S+[.com]','',cleantext)
            rem_special = re.sub(r'[_"\-;%()|~^+&=*%.,!?:#$@\[\]/]', ' ', rem_email)    
            rem_url=re.sub(r'http\S+', '',rem_special)
            rem_num = re.sub('[0-9]+', '', rem_url)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(rem_num)  
            filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
            # filtered_words = [w for w in tokens if not w in stopwords.words('english')]
            stem_words=[stemmer.stem(w) for w in filtered_words]
            lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
            return " ".join(filtered_words)
        
        dfs['cleaned_text']=dfs['Extracted_text'].apply(preprocess)

        for i in dfs['cleaned_text']:
            # print("^^^^^^^^^^^^^^^^",i)
            new_data = "'"+i+"'"
            # print("9090900900909",new_data)
            new_list=[word for word in temp if word in new_data]
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^",new_list)
            # print("$$$$$$$$$$$$$$$$",f)
            counterA = Counter(temp)
            counterB = Counter(new_list)
            def counter_cosine_similarity(c1, c2):
                terms = set(c1).union(c2)
                dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
                magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
                magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
                print("------------",magB)
                if magB==0.0:
                    result=0.0
                else:
                    result = dotprod / (magA * magB)
                return result
            maching_percentage=(counter_cosine_similarity(counterA, counterB) * 100)
                # print("@@@@@@@@@@@@@@",maching_percentage)
            similarity_percentage.append(maching_percentage)
        
        # dfs['similarity_percentage']=similarity_percentage
        dfs['similarity_percentage']= ["%.2f" % elem for elem in similarity_percentage]
        print("::::::",dfs)
        df=dfs[['Names','similarity_percentage','Skills']]


        df=df.sort_values(by='similarity_percentage', ascending=False).reset_index(drop=True)
        print("-------all_df-------",df)
        data_list = [] 
        for index, row in df.iterrows():
            print(row['Names'], row['similarity_percentage'])
            XX={"name":row['Names'],"Similarity_Percent":row['similarity_percentage'],"Skills":row['Skills']}
            print("********",XX)
            data_list.append(XX)
            serializer = CV_ScreenSerializer(data=XX)
            print("------------------Serlizer__------------",serializer)
            if serializer.is_valid():
                serializer.save()
            else:
                print(serializer.errors)


        context={'data': data_list}   
        return render(request,'index.html',context)
