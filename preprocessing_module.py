import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## Defining GLOBALS ######################################################################################################
HAM_DIR = './Dataset/ham/'
SPAM_DIR = './Dataset/spam/'
HAM_CLEAN_DIR = './Dataset/ham_clean/'
SPAM_CLEAN_DIR = './Dataset/spam_clean/'
FEATURES_FILE_DIR = './Dataset/features.txt'
ENG_VOCAB_FEATURES_FILE_DIR = './Dataset/eng_vocab_features.txt'
CLEANR_HTML = re.compile('(<.*?>|&[a-zA-Z]+;)')
URL = re.compile("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
EMAIL = re.compile('[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}')
NUMBER = re.compile('[0-9]+')
DOLLAR = re.compile("\$")
SPACE = re.compile("( |\t)+")
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()
eng_stop_words = stopwords.words('english')
#########################################################################################################################

def walk_till_empty(lines , i):
    """
    Returns the index of the first line that is not empty starting from i.

            Parameters:
                    lines (list(string)): the list of lines
                    i (int): The starting index

            Returns:
                    index of the first line that is not empty starting from i.
    """
    while i+1 < len(lines) and lines[i]!="\n":
       i+=1
    return i   

def extract_email(filename):
    """
    Returns the email contained in the given file (ignoring the headers).

            Parameters:
                    filename string : the file name

            Returns:
                    the email contained in the given file.
    """
    with open(filename,encoding = "ISO-8859-1") as f:
           lines = f.readlines()
    i = walk_till_empty(lines, 0)
    while len(lines[i+1].strip().split(" ")[0].lower()) == 0 or (lines[i+1].strip().split(" ")[0].lower()[0] >= 'a' and 
    lines[i+1].strip().split(" ")[0].lower()[0] <= 'z' and 
    (not re.match(URL,lines[i+1]) and lines[i+1].strip().split(" ")[0].find(":") != -1)):
    
        i = walk_till_empty(lines, i+1)

    return lines[i+1:]

def create_ham_clean():
    """
    Creates clean ham emails .
    """
    for _ ,__ ,ham_files in os.walk(HAM_DIR):
        print("ham files found : ",len(ham_files))
    n = 0
    for ham_file in ham_files :
        outF = open(f"{HAM_CLEAN_DIR}ham{n}.txt", "w",encoding = "ISO-8859-1")
        n+=1
        glob = ""
        for line in extract_email(f"{HAM_DIR}{ham_file}"):
            if(not len(line.strip()) == 0):
                glob+=" "+line
        
        line = re.sub(SPACE," ",glob.lower().replace("\n"," ").strip())
        
        line = re.sub(CLEANR_HTML, '', line)
        line = re.sub(URL,"httpaddr",line)
        line = re.sub(EMAIL,"emailaddr",line)
        line = re.sub(NUMBER,"number",line)
        line = re.sub(DOLLAR,"dollar",line)
        line = [lemmatizer.lemmatize(word) for word in word_tokenize(line) if word not in eng_stop_words]
        line = " ".join(line)
        line = line.translate(str.maketrans('', '', string.punctuation))
        print(line,file=outF)

        outF.close()


def create_spam_clean():
    """
    Creates clean spam emails .
    """
    for _ ,__ ,spam_files in os.walk(SPAM_DIR):
        print("spam files found : ",len(spam_files))
    n = 0
    for spam_file in spam_files :
        outF = open(f"{SPAM_CLEAN_DIR}spam{n}.txt", "w",encoding = "ISO-8859-1")
        n+=1
        glob = ""
        try :
            mail = extract_email(f"{SPAM_DIR}{spam_file}")
            for line in mail:
                if(not len(line.strip()) == 0):
                   glob+=" "+line   
        except Exception:
            print(spam_file)
        line = re.sub(SPACE," ",glob.lower().replace("\n"," ").strip())
        line = re.sub(CLEANR_HTML, '', line)
        line = re.sub(URL,"httpaddr",line)
        line = re.sub(EMAIL,"emailaddr",line)
        line = re.sub(NUMBER,"number",line)
        line = re.sub(DOLLAR,"dollar",line)
        line = [lemmatizer.lemmatize(word) for word in word_tokenize(line) if word not in eng_stop_words]
        line = " ".join(line)
        line = line.translate(str.maketrans('', '', string.punctuation))
        print(line,file=outF)
        outF.close()

def create_vocab(K):
    """
    Returns the vocab used in our clasefication.

            Parameters:
                    k (int) : the minimum number of time a word was used in the emails

            Returns:
                    the vocab dictionnary.
    """
    vocab = {}
    for _ ,__ ,clean_spam_files in os.walk(SPAM_CLEAN_DIR):
        print("clean spam files found : ",len(clean_spam_files))
    for clean_spam_file in clean_spam_files :
        with open(SPAM_CLEAN_DIR+clean_spam_file,encoding = "ISO-8859-1") as f:
           lines = f.readlines()
        line = lines[0]
        tokenized = word_tokenize(line)
        for word in tokenized :
            if word not in vocab.keys() :
                vocab[word] = 1
            else :
                vocab[word]+= 1
    for _ ,__ ,clean_ham_files in os.walk(HAM_CLEAN_DIR):
        print("clean ham files found : ",len(clean_ham_files))
    for clean_ham_file in clean_ham_files :
        with open(HAM_CLEAN_DIR+clean_ham_file,encoding = "ISO-8859-1") as f:
           lines = f.readlines()
        line = lines[0]
        tokenized = word_tokenize(line)
        for word in tokenized :
            if word not in vocab.keys() :
                vocab[word] = 1
            else :
                vocab[word]+= 1
    return {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1],reverse=True) if v > K and k in english_vocab and len(k)>=2}
        

    
