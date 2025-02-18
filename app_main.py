import argparse
from app_class import *
import pandas as pd
from sklearn.model_selection import train_test_split
from app_model import *

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Ruta del dataset", required=True)
    parser.add_argument("--use_lema", type=bool, help="Lematizar los Tokens [default=True]", required=False)
    parser.add_argument("--use_stop_words", type=bool, help="Eliminar Stop Words [default=True]", required=False)
    parser.add_argument("--type_model", type=str, help="Tipo de modelo ['TF-IDF', 'CountVectorizer', 'all']", required=True)
    args = parser.parse_args()
    return args

def load_dataset(path):
    df_train=pd.read_csv(path)
    x_train, x_test, y_train, y_test = train_test_split(df_train['review'], df_train['sentiment'], test_size=0.2, random_state=42, stratify=df_train['sentiment'])
    return x_train, x_test, y_train, y_test

def main():
    print('Inicializando....')
    args_values=get_arguments()
    x_train, x_val, y_train, y_val= load_dataset(args_values.data_path)
    if args_values.use_lema is None:
        args_values.use_lema=True
    if args_values.use_stop_words is None:
        args_values.use_stop_words=True    
    tokenizer=Tokenizer(args_values.use_lema,args_values.use_stop_words)
    print('Procesando Datasets....')
    x_train_tokenized = tokenizer.preprocess_texts(x_train,64)
    x_val_tokenized = tokenizer.preprocess_texts(x_val,64)
    x_train_joined = [" ".join(doc) for doc in x_train_tokenized]
    x_val_joined = [" ".join(doc) for doc in x_val_tokenized]
    model=MModel(x_train_joined,y_train,x_val_joined,y_val,args_values.type_model)
    model.train_()
    
    
    
if __name__ == '__main__':
    main()    