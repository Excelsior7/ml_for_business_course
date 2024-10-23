import pandas as pd
from pydantic import BaseModel
import re
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# UTILITIES
def load_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
def join_embeddings_to_df(jobs_data,augmented_description_embeddings):
    jobs_data = jobs_data.reset_index(drop=True)
    columns_embeddings_df = [str(c) for c in range(augmented_description_embeddings.shape[1])]
    return pd.concat([jobs_data,pd.DataFrame(augmented_description_embeddings, columns=columns_embeddings_df)],axis=1,join='inner')


# CLASSES
class ModelInput(BaseModel):
    remote_allowed: int
    work_type_contract: bool
    work_type_full_time: bool
    work_type_part_time: bool
    state: str
    company_name: str
    title: str
    description: str

    def df_create(self) -> pd.DataFrame:
        columns = ["remote_allowed"
                   ,"work_type_CONTRACT"
                   ,"work_type_FULL_TIME"
                   ,"work_type_PART_TIME"
                   ,"state"
                   ,"company_name"
                   ,"title"
                   ,"description"]
        
        df = pd.DataFrame([[self.remote_allowed
                            ,self.work_type_contract
                            ,self.work_type_full_time
                            ,self.work_type_part_time
                            ,self.state
                            ,self.company_name
                            ,self.title
                            ,self.description]], columns=columns)
        return self.df_preprocessing(df)
    

    def clean_description(self,text):
        stop_words_path = Path("stop_words.pkl")
        stop_words = load_file(stop_words_path)
        text = re.sub(r'\W', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    
    def df_preprocessing(self, df):
        COLUMNS_TO_CONCATENATE = ['company_name', 'title', 'description']
        df["title"] = df["title"].str.strip()
        df[COLUMNS_TO_CONCATENATE] = df[COLUMNS_TO_CONCATENATE].fillna("-",)
        df["augmented_description"] =  df[COLUMNS_TO_CONCATENATE].agg(' '.join, axis=1)
        df = df.drop(columns=["company_name","title","description"])
        df['augmented_description'] = df['augmented_description'].apply(self.clean_description)
        return self.augmented_description_embedding(df)
    
    def augmented_description_embedding(self,df):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        augmented_description_list = df["augmented_description"].to_list()
        augmented_description_embeddings = model.encode(augmented_description_list)
        df = df.drop(columns=["augmented_description"])
        df = join_embeddings_to_df(df,augmented_description_embeddings)
        return df


class ModelOutput(BaseModel):
    salary: float