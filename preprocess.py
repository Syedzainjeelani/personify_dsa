#Checkout if the imports are needed here too ?
#Imports...

import re
import nltk
wn = nltk.stem.WordNetLemmatizer()
pstemmer = nltk.stem.PorterStemmer()
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download() # to choose any pkj to download
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
import contractions
# from tqdm import tqdm
from numba import njit, cuda, jit  
# from numbapro import cuda
#===========================

class CleanText():
  
  def __init__(self, text = "test"):
        self.text = text
        # self.words = words


#remove any links from each post row of 50 texts
  def remove_links(self):
      no_link_text = re.sub(r"http\S+", "", self.text, flags=re.MULTILINE)
      # no_link_text = re.sub(r"[|||]", "", no_link_text, flags=re.MULTILINE)
      list_sent = no_link_text.split("|||")
      no_link_text = " ".join(list_sent)
      # print("Links Removed.")
      self.text = no_link_text
      return self

  def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)
        return self

  def replace_contractions(self):
        """Replace contractions in string of text"""
        self.text = contractions.fix(self.text)
        return self

  #Takes in 'without-links' text and divides them into countable words
  def tokenize_text(self):
      word_tokens = word_tokenize(self.text)
      # self.words = " ".join(word_tokens)
      self.words = word_tokens
      # print("Words Tokenized.")
      return self

  def to_lowercase(self):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        # new_words = ""
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
            # new_word += f"{new_word} "
        self.words = new_words
        return self

  def remove_punctuation(self):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        # new_words = ""
        for word in self.words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
                # new_words += f"{new_word} "
        self.words = new_words
        return self


  def remove_ptype_words(self):
    """Remove personality types words given in text from list of tokenized words"""
    new_words = []
      # new_words = ""
    for word in self.words:
        #after making words to lower letters
      new_word = re.sub(r'\b(istj|istp|isfj|isfp|infj|infp|intj|intp|estp|estj|esfp|esfj|enfp|enfj|entp|entj)', '', word)
      if new_word != '':
        new_words.append(new_word)
                # new_words += f"{new_word} "
    self.words = new_words
    return self

  #Remove any stop words from tokenized text
  def remove_stop_words(self):
    stopwords_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    text_tokens = []
    # text_tokens = ""
    for word in self.words:
      if not word in stopwords_list:
        text_tokens.append(word)
              # text_tokens += f"{word} "
    self.words = text_tokens
      # print("Stop Words Removed.")
    return self

  #Convert any family of words to their root word ( like walking , walked -> walk)
  def stem_words(self):
      # lemm_text = ""
    stem_text = []
    for word in self.words:
        # stem_text += f"{pstemmer.stem(word)} " 
      stem_text.append(pstemmer.stem(word)) 
    self.words = stem_text
      # print("Text Lemmatized.")
    return self
      
  def lemmatize_verbs(self):
    """Lemmatize verbs in list of tokenized words"""
    lemmas = []
    # lemmas = ""
    for word in self.words:
      lemma = wn.lemmatize(word, pos='v')
      lemmas.append(lemma)
            # lemmas += f"{lemma} "
    self.words = lemmas
    return self
 
  def join_words(self):
    self.words = ' '.join(self.words)
    return self
  
  # @cuda.jit(device=True)
  # @njit(parallel=True)
  @jit(parallel=True)
  def preprocess_text(self, text):
    self.text = text
    # with tqdm(total=len(text)) as pbar:
      #Normalization
    # self = self.remove_links()
    # self = self.remove_numbers()
    # self = self.replace_contractions()

    #Working with words
    self = self.tokenize_text()
    # self = self.to_lowercase()
    # self = self.remove_punctuation()

    # #Stemmize, lemmatize and join words back to one text
    # self = self.remove_ptype_words()
    self = self.remove_stop_words()
    
    #Words rooting
    # self = self.stem_words()
    # self = self.lemmatize_verbs()
    self = self.join_words()
    # pbar.update(1)
    
    # print("preprocess ends.")
    return self.words
