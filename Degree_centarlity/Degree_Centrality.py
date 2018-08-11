import sys
import math
import nltk.data
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rouge import Rouge
DEFINE_THRESHOLD = 0.20
DEFINE_FILEPATH='Topic2/'
# DEFINE_GOLDSTANDARD='Topic1.1/'

specialChars = [',',':','?','#','$','.','!','@','^','*','+']
uniqueWords = set()  #Preprocessed Words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sentenceDictionary = {}
idfDictionary = {}
sentenceVector = {}
cosineMatrix = {}
cosineList = {}
cosineListLen = {}
new_cosineListLen = {}
word_count=0
word_limit=250
def Data_Preprocessing():
    try:
      proFile = open (DEFINE_FILEPATH+'processed','w')
      originalLineList = []
      for filename in os.listdir(DEFINE_FILEPATH):
        fp=open(DEFINE_FILEPATH+filename,"r")
        lines=fp.readlines()
        cleanPattern = re.compile('<[^>]+>')
        for line in lines:
            line=line.strip()
            line=re.sub(cleanPattern, '', line)
            if ( line != '' ):
	     proFile.write(line)
      proFile.close() 
    except IOError:
           print "File does not exist." 
     
 

def Creating_Index():
    global sentenceDictionary
    sentIndex = 0
    try:
        file = open(DEFINE_FILEPATH+'processed','r')
        text = file.read()
        listOfSentences = sent_tokenize(text,language='english')
        for sentence in listOfSentences:
            sentenceDictionary.update({sentIndex:sentence})
            sentIndex = sentIndex + 1
        file.close()
    except IOError:
    		print "mandela2"
        	print 'File does not exist.'


def rougeScore():
	try:
		fileReadC = open(DEFINE_FILEPATH+'summary.txt','r')
		reference = fileReadC.read()
		fileRead = open(DEFINE_FILEPATH + 'Topic2.1','r') 
		text = fileRead.read()
		rouge = Rouge()
		print rouge.get_scores(reference, text)
	except IOError:
			print "mandela1"
			print "File does not exist!!" 
def write_into_files(line):
	global word_count
	# print line
	word_separated_line=word_tokenize(line)
	for word in word_separated_line:
		word_count+=1
	if word_count<=word_limit:
		with open(DEFINE_FILEPATH+'summary.txt', 'a+') as the_file:
			the_file.write(line)
	else:
		return 1
def wordPreprocessing ( word ):   #Preforms word preprocessing
    word = word.lower()  # Case Unfolding
    word = lemmatizer.lemmatize(word) #Lemmatization
    return word


def Unique_Word_Finding ( sentenceDictionary ):  #Finds Distinct words
    global uniqueWords
    listOfKeys = sentenceDictionary.keys()
    listOfKeys.sort()
    for key in listOfKeys:
               sentence = sentenceDictionary.get(key)
               listOfWords =  word_tokenize(sentence)
               listOfWords = [word for word in listOfWords if not word in stop_words]
               listOfWords = [word for word in listOfWords if not word in specialChars]
               for word in listOfWords:
                   wordNew = wordPreprocessing ( word )
                   uniqueWords.add(wordNew)
    print  len(uniqueWords) #test              
    

def calculateIDF( uniqueWords, sentenceDictionary ):  #Calculates IDF
    global idfDictionary 
    bagOfWords = len(uniqueWords)
    listOfKeys = sentenceDictionary.keys()
    listOfKeys.sort()
    for uniqueWord in uniqueWords:
       wordCount = 0
       for key in listOfKeys:     
               sentence = sentenceDictionary.get(key)
               listOfWords =  word_tokenize(sentence)
               listOfWords = [word for word in listOfWords if not word in stop_words]
               listOfWords = [word for word in listOfWords if not word in specialChars]
               for word in listOfWords: 
                   wordNew = wordPreprocessing(word)
                   if (wordNew == uniqueWord ):
                      wordCount = wordCount + 1
       idf = (math.log(float(bagOfWords)/float(wordCount))/math.log(2.0)) 
       idfDictionary.update({uniqueWord:idf})
    
def Creating_Vectors (uniqueWords, sentenceDictionary, idfDictionary):  #Creates Sentence vectors
    global sentenceVector
    listOfKeys = sentenceDictionary.keys()
    listOfKeys.sort()
    for key in listOfKeys:
        sentence = sentenceDictionary.get(key)
        listOfWords =  word_tokenize(sentence)
        listOfWords = [word for word in listOfWords if not word in stop_words]
        listOfWords = [word for word in listOfWords if not word in specialChars]
        vectorValues = []
        for uniqueWord in uniqueWords:
            termFrequency = 0
            for word in listOfWords: 
                wordNew = wordPreprocessing ( word )
                if ( uniqueWord == wordNew ): 
                         termFrequency = termFrequency + 1
            idf = idfDictionary.get(uniqueWord)
            vectorValues.append( termFrequency * idf )
        sentenceVector.update({key:vectorValues}) 
    
def calculateCosineSimilarity(sentenceVector,outerKey,innerKey):  #Caculates Cosine Similarity
    sentOneVector = sentenceVector.get(outerKey)
    sentTwoVector = sentenceVector.get(innerKey)
    vectorSum = 0.0
    sentOneRoot = 0.0
    sentTwoRoot = 0.0
    for value in sentOneVector:
        sentOneRoot = sentOneRoot + (value*value)
    for value in sentTwoVector:
        sentTwoRoot = sentTwoRoot + (value*value)
    sentOneRoot = float(math.sqrt(sentOneRoot))
    sentTwoRoot = float(math.sqrt(sentTwoRoot))
    for iterator in range(0,len(sentOneVector)):
              vectorSum = vectorSum + ( sentOneVector[iterator]*sentTwoVector[iterator] )
    return (float(vectorSum)/float((sentOneRoot*sentTwoRoot)))

def Creating_Cosine_Matrix ( sentenceVector ):  #Builds Cosine Matrix
     global cosineMatrix
     global cosineList
     listOfKeys = sentenceVector.keys()
     listOfKeys.sort()
     for outerKey in listOfKeys:
         adjacentNodes = []
         adjacentNodes_2 = []
         adjacentNodes_3 = []
         for innerKey in listOfKeys:
             similarity = calculateCosineSimilarity(sentenceVector,outerKey,innerKey)
             if ( similarity >= DEFINE_THRESHOLD ):         
                  adjacentNodes.append(similarity)
                  adjacentNodes_2.append(innerKey)
             else:
                  adjacentNodes.append(-1)
         degreeOfNodes = 0
         for item in adjacentNodes:
             if (item != (-1)):
                degreeOfNodes = degreeOfNodes + 1
         adjacentNodes.append(degreeOfNodes)
         cosineMatrix.update({outerKey:adjacentNodes})
         cosineList.update({outerKey:adjacentNodes_2})
         cosineListLen.update({outerKey:len(adjacentNodes_2)})
    	

     
def main():
  Data_Preprocessing()
  Creating_Index()
  Unique_Word_Finding(sentenceDictionary) 
  calculateIDF( uniqueWords, sentenceDictionary ) 
  Creating_Vectors (uniqueWords, sentenceDictionary, idfDictionary)  
  Creating_Cosine_Matrix ( sentenceVector )
  # print cosineList.sort()
  summary_sentence=[]
  discarded_sentence=set()
  list_A=sorted(cosineListLen.items(), key=lambda x: x[1],reverse=True)
  # print list_A[0][0]
  for i in range(len(list_A)):
  	max_key=list_A[i][0]
  	if max_key not in discarded_sentence:
  		summary_sentence.append(max_key)
  		temp_discarded_list=cosineList[max_key]
  		for i in temp_discarded_list:
  			discarded_sentence.add(i)

  for sentence in summary_sentence:
  	flag=write_into_files(sentenceDictionary[sentence])
  	if flag==1:
  		break
  rougeScore()



main()
          




        

                


