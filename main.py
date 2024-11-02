import pandas as pd
import numpy as np
import re
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords',quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('punkt',quiet=True)

from thefuzz import fuzz
from thefuzz import process

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from scipy.sparse import csr_matrix


def data_pre_pre_processing(df:pd.DataFrame):
    df= (df[["wonum","description","ldtext","mats_assigned","wopriority","actstart"]]
         .drop_duplicates()
         .reset_index(drop=True)
    )
    df[["ldtext", "description"]] = df[["ldtext", "description"]].astype(str)
    df["actstart"] = pd.to_datetime(df["actstart"])

    un_wonums=(df[["wonum"]].drop_duplicates(keep=False))
    df=un_wonums.merge(df,on='wonum').reset_index(drop=True)

    return df


def text_pre_processing(text):

    # Remove numbers and punctuation
    clean_text = "".join([i for i in text if i.isalpha() or i.isspace()])
    # Remove exceess whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # Transform to lower case
    clean_text = clean_text.lower()

    tokens = nltk.word_tokenize(clean_text)
    #Removestopwords and character-like words
    clean_tokens = [w for w in tokens if (not w in stopwords.words("english")) and (len(w) > 2)]

    # Lemmatizatize the words(not stemming as we will use doc2vec later on which captures the meaning of words, therefore stemming is not applicable in this case)
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in clean_tokens]

    return lemm_text


def gather_relevant_mats(columns, df):

    def a2v_extraction(text):

        if text != 'nan':
            # Remove exceess whitespace
            clean_text = re.sub(r'\s+', ' ', text)
            pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'
            # Replace matched punctuation with a space
            clean_text = re.sub(pattern, ' ', clean_text)
            tokens = nltk.word_tokenize(clean_text)

            a2v_nums = [(w)
                        for w in tokens if w.startswith('A2V') and len(w) > 5]
        else:
            return ''

        return ",".join(list(set(a2v_nums)))

    mats_cols = []
    for col in columns:
        df[f'mats_from_{col}'] = df[col].apply(a2v_extraction)
        mats_cols.append(f'mats_from_{col}')

    df['all_relevant_mats'] = (df[mats_cols].astype(str).apply(','.join, axis=1)
                                            .apply(lambda x: list(set(x.split(','))))
                               )
    df = df.drop(columns=mats_cols)

    return df

def data_sanitising(df: pd.DataFrame):

    df["actstart"] = pd.to_datetime(df["actstart"])
    df["clean_ldtext_size"] = df["clean_ldtext"].apply(lambda x: len(x))
    df["clean_description_size"] = df["clean_description"].apply(lambda x: len(x))

    # Fix data quality
    df['clean_description'] = df['clean_description'].fillna('[]')
    df['clean_ldtext'] = df['clean_ldtext'].fillna('[]')
    df['ldtext'] = df['ldtext'].astype(str)
    df['description'] = df['description'].astype(str)
    df['clean_ldtext'] = df['clean_ldtext'].astype(str)
    df['clean_description'] = df['clean_description'].astype(str)
    
    df = df.replace({"['nan']": '[]', '[nan]': '[]'})
    df = df.drop(df[df["ldtext"].astype(str).apply(lambda x: x.startswith('[if gte mso 9]&'))].index,axis=0).reset_index(drop=True)

    cols = ['description', 'ldtext']
    df = gather_relevant_mats(cols, df) 
        
    return df


def exact_matches(df):
    
    df1 = df.groupby(["clean_description", "clean_ldtext"], dropna = False).count()
    df1 = (df1[df1["wonum"] > 1]["wonum"].reset_index(drop=False))
    df1 = df1.rename(columns={"wonum": "count"}).reset_index(drop=True)
    df1["group_id"] = df1.index 
    df = df.merge(df1,on= ["clean_description", "clean_ldtext"], how='left')
    
    group_map = (df.groupby("group_id", dropna=True)['wonum'].apply(lambda x: ",".join(x))
                 .reset_index()
                 .rename(columns={"wonum": "similar_clean_description_ldtext"})
                )
    df = df.merge(group_map, on=["group_id"], how='left')
    
    df["exact_match"] = np.where(df["group_id"].isna(),0,1)
    
    return df


def tf_idf_similarity_df(df:pd.DataFrame, col: str, vect_max_feats : int =500 ,n_splits: int = 10):
      
      if n_splits <= 0 or n_splits > len(df):
        raise ValueError(
            "n_splits should be a positive integer less than or equal to the length of the dataframe")

      q_sim_df = pd.DataFrame()

      corpus = [" ".join( (ast.literal_eval(text))) for text in df[col][0:int(len(df)/n_splits)]]
      vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features = vect_max_feats)
      X = vectorizer.fit_transform(corpus).astype(np.float32)

      print(f'We have {len(corpus)} documents and {len(set(vectorizer.get_feature_names_out()))} unique words in our corpus.\n'
            f'Tf-idf matrix is a {X,X.dtype}') 

      
      sim_mat = cosine_similarity(X)#.astype(np.float32)
      q_sim_df = pd.DataFrame.sparse.from_spmatrix(csr_matrix(np.round(sim_mat.data, 2)))
      
      return q_sim_df


def get_cluster_info(des_mat: pd.DataFrame, df: pd.DataFrame, key_word: str, similarity: float):
    
      words = key_word.split(" ")
      # " ".join(text_pre_processing(key_word))
      matches = df[df["clean_description"].apply(lambda x: all(word in ast.literal_eval(x) for word in words))]

      if len(matches) != 0:
            print(f'Which description best matches what you are looking for?\n')
            print(matches['description'].head(50))
      else:
            print('No matches found in this dataset')
      
      key = input()

      return df.iloc[des_mat[des_mat[int(key)] > similarity].sort_values( int(key), ascending=False).index]


def get_similar_wonums(des_mat: pd.DataFrame, df: pd.DataFrame, index_key: int, similarity: float):

      matches = df.iloc[des_mat[des_mat[int(index_key)] > similarity].sort_values(int(index_key), ascending=False).index]
     
      if (len(matches) != 0) and (len(matches) != 1):
          print(f'Descriptions matched:\n')
          print(matches['description'].drop_duplicates().head(50))

      else:
          print('No matches found in this dataset')
          return None

      return ",".join(matches['wonum'])


if __name__ =='__main__':

    df = pd.read_csv(r"brake_workorders.csv")
    df=data_pre_pre_processing(df)
    df["clean_ldtext"] = df['ldtext'].apply(lambda x: text_pre_processing(x))
    df["clean_description"] = df['description'].apply(lambda x: text_pre_processing(x))
    
    non_exact_ld_matches_df = df[(df["exact_match"] == 0)].reset_index(drop= True)

    n = len(non_exact_ld_matches_df) 
    q_sim_df = tf_idf_similarity_df(df, col='clean_description', vect_max_feats=1000, n_splits=1)
    similarity = 0.75
    get_similar_wonums(q_sim_df, df, index_key=0, similarity=0.75)
    get_cluster_info(q_sim_df,df,'brake',similarity=0.8)