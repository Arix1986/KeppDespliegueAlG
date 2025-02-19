

import re
import contractions
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import unicodedata
stop_words_sklearn = set(ENGLISH_STOP_WORDS)


class Tokenizer:
    def __init__(self, use_lemma=True, use_stw=True):
        self.nlp = spacy.load("en_core_web_md")
        self.use_lemma = use_lemma
        self.sw_list = stop_words_sklearn if use_stw else set()
        
    def preprocess_texts(self, texts, batch_size):
        processed_texts = []
        stopword_exceptions = {"not", "no", "yes", "never", "hardly", "barely", "only", "even", "just"}

        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = []
            
            for token in doc:
                token_text = token.lemma_.lower() if self.use_lemma else token.norm_.lower()
                token_text = contractions.fix(token_text)
                token_text = re.sub(r"[^\x00-\x7F]+", " ", token_text)
                token_text = re.sub(r"[^\w\s]", "", token_text) 
               
            
                if token_text not in stopword_exceptions and (token.is_stop or token_text in self.sw_list):
                    continue

            
                          
                if not token_text.strip() or token_text.isnumeric() or \
                re.match(r"(https?://\S+|www\.\S+)", token_text) or \
                re.match(r"\S+@\S+\.\S+", token_text):
                    continue 

                tokens.append(token_text)
            
            processed_texts.append(tokens)   
        return processed_texts


