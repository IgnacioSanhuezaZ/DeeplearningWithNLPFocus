# Importing necessary libraries/modules; requires to be executed once for every session
import numpy as np
import matplotlib.pyplot as plt
import os
from random import shuffle
import re
import nltk

# from bokeh.models import ColumnDataSource, LabelSet
# from bokeh.plotting import figure, show, output_file
# from bokeh.io import output_notebook
from wordcloud import WordCloud

import zipfile
import lxml.etree

# output_notebook()
nltk.download('punkt')

# Upload the dataset if it's not already there: this may take a minute..
file_name = './dataSet/ted_en-20160408.zip'
counter = 0
while not os.path.isfile(file_name) and counter <= 5:
    """google models for connecting with google colab not suported with python 3.9 as intended so I just commented it"""
    # from google.models import files
    # select the file "ted_en-20160408.zip" from your local drive here
    print("invalid file path!")
    file_name = input("enter new path: \n")
    counter += 1
if counter >= 5:
    print("Too many retries. Exiting the app...")
else:
    print("name and adress of file found in ", file_name)

# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile(file_name, 'r') as z:
    new_parsed_name = file_name.split('/')
    print(new_parsed_name[2].split(".")[0])
    xml_name = new_parsed_name[2].split(".")[0] + '.xml'
    print(xml_name)
    doc = lxml.etree.parse(z.open(xml_name, 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))

print(type(input_text))

# Extract all the tags in the XML
tags = [element.tag for element in doc.iter()]

# Get unique tags using a set
unique_tags = set(tags)

# Print the unique tags
for tag in unique_tags:
    print(tag)

# Delete the variable doc to save space as we have alreay extracted the necessary data we need.
del doc

'''
The following part of the code shows a chunk of text from our ted text dataset. 
Have a look and try to identify three issues you can think of that can create a problem for text analysis, next to the 
one which is already provided.
When giving your answer, we ask you to also mention why the issues could be problematic (similar to the example given).
Each correct answer (naming + explaining the problem) will give 1 point.
'''

# Have a look at the output of this code, to see some text examples.
i = input_text.find("Hyowon Gweon: See this?")
print(input_text[i:i + 300])
print()

i = input_text.find("You will earn")
print(input_text[i:i + 245])

'''
Your Solution goes here:
- Speakers' names: embeddings for names will dominate the embedding space unnecessarily.
- Type of resource: the type of resource would be irrelevant for the word analysis so it also uses more space 
                    than necessary
- Action descriptions: the description of the actions are also irrelevant for the task regarding only wordings
- It has too many expression and quotation marks: those symbols does not provide valuable information to the task 
                                                  asked for in this case
'''
'''
Next we want to create a preprocessing pipeline to later clean the entire dataset in one go. 
The pipeline takes input_text as input and should provide a cleaned and ready-to-use text data called cleaned_text.

Your task is to implement this pipeline with three functions that each take care of one of the three issues you listed 
in exercise 1.1.

Some hints about the pipeline are given as well as the code for the example from 1.1.
'''


def remove_speaker(text):
    ''' takes the text as an input and removes the name of the speaker as output '''

    text_to_work_with = text
    x = []
    for line in text_to_work_with.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        x.extend(m.groupdict()['postcolon'])
    without_speaker = "".join(x)
    return without_speaker


# To Do ##
# implement your 3 functions.
# Make the name of the functions sensible.
def remove_resource_type(text):
    text_to_work_with = text
    m = ""
    for line in text_to_work_with.split('\n'):
        m += re.sub(r'\([^)]*\)', ' ', line)
        # x.extend(m.groupdict()['postcolon'])
    resource_free_text = "".join(m)
    return resource_free_text


def remove_symbols(text):
    text_to_work_with = text
    m = ""
    for line in text_to_work_with.split('\n'):
        m += re.sub('[^a-zA-Z0-9.]+', ' ', line)
    symbol_free_text = "".join(m)
    return symbol_free_text


def remove_action_descriptions(text):
    return remove_resource_type(text)


def text_cleaned(text):
    ''' takes the raw text as input. Runs the text through cleaning functions.
       outputs a clean an preprocessed text for further analysis. '''

    ## To Do ##
    # include your functions here - you can order the pipeline however you want.
    text_no_speaker = remove_speaker(text=text)
    text_no_parentesis_content = remove_action_descriptions(text=text_no_speaker)
    cleaned_text = remove_symbols(text=text_no_parentesis_content)
    return cleaned_text


# print("lenght of input text: ", len(input_text))
input_text_clean = text_cleaned(input_text)
# print("lenght of cleaned text: ", len(input_text_clean))
# i = input_text_clean.find("See this")
# print(input_text_clean[i:i + 300])
# print()
#
# i = input_text_clean.find("You will earn")
# print(input_text_clean[i:i + 245])

'''
Exercise 1.3 (6 Points)
To continue with building our embedding, we need to tokenize every single word (so that the model has individual 
tokens to process). Therefore we first need to split the text into sentences and after that into words. 
Try it yourselves or use the NLTK-Tools build for this 
(https://www.kite.com/python/docs/nltk.word_tokenize + https://www.kite.com/python/docs/nltk.sent_tokenize). 
To make it easier, we should also delete every character that is not a letter. Additionally, we could decrease the size 
of our vocabulary. A way to do this is by converting capital characters to lower case characters 
(but it also has some drawbacks - more on this in exercise 1.4).

Split your text into sentences and save them in the array sentences_strings_ted. 
Save one variabale tokens with all the tokens in the text and one array named sentences_ted that contains an array for 
every sentence, with all the tokenized words of that sentence.

Example:
If the text looks like this: "I love cake. You have to be honest, you love it too!", the variables should look like:
sentences_strings_ted=['I love cake.', 'You have to be honest, you love it too!']
sentences_ted=[['i', 'love', 'cake'], ['you', 'have', 'to', 'be', 'honest', 'you', 'love', 'it', 'too']]
tokens=['i', 'love', 'cake', 'you', 'have', 'to', 'be', 'honest', 'you', 'love', 'it', 'too']


IMPORTANT: Apply this to input_text_clean.


[Hint:] use pickle file (.pkl) to dump and load the variables like sentences_strings_ted, tokens, 
sentences_ted to continue where you left, when you comeback next time. It will save a lot of time/effort.
'''

sentences_strings_ted = input_text_clean.split(".")
print("sentences_string_ted[:10]: ", sentences_strings_ted[:10])
sentences_ted = []
for s in sentences_strings_ted:
    if s and s[0] == ' ':
        sentences_ted.append(s[1:].split(" "))
    elif s:
        sentences_ted.append(s.split(" "))
    if sentences_ted[-1][-1] and sentences_ted[-1][-1] == '':
        sentences_ted[-1].pop(-1)
# print("sentences_ted[:10]: ", sentences_ted[:50])
tokens = [x for flaten_s in sentences_ted for x in flaten_s]
# print("tokens[:90]: ", tokens[:90])


'''
Exercise 1.4 (1 Point)
The good side of converting all capital letters is, that we reduce the volume of the vocabulary. 
Thereby we dont differentiate between the the words "today" and "Today". But there is a caveat. 
Can you think of any downside to this process?

Your answer goes here:

There are semantic differences that could show up at analysing the meaning of each word at vectorizing in 
the next steps that wont take the type of usage of the word or the semantic function of the words into account

Let's quickly see how large our vocabulary turned out to be!
'''

'''
Exercise 2.1 (2 Points)
Your next task will be to store the counts of the top 1,000 most frequent words in a list called counts_ted_top1000 ! 
There are multiple ways to do this. You can have a look at the Counter-Function 
(https://docs.python.org/2/library/collections.html) or the FreqDist-Function 
(https://www.kite.com/python/docs/nltk.FreqDist). If you don't trust any of these, you can of course build your own 
function. In the end we want an array with tuples of the structure:

counts_ted_top1000 = [(WordA,FrequencyA),(WordB,FrequencyB)]
'''

# Your code goes here

mostfreqn = 30  # Here we define how many of them we want to see in the diagramm
frequency = [y for (x, y) in counts_ted_top1000][:mostfreqn]
word = [x for (x, y) in counts_ted_top1000][:mostfreqn]
indices = np.arange(len(counts_ted_top1000[:mostfreqn]))
plt.bar(indices, frequency, color='r')
plt.xticks(indices, word, rotation='vertical')
plt.tight_layout()
plt.show()

'''
Exercise: You can clearly see, that many of the most common words are redundant and not very meaningful. 
These types of words are called stopwords. What problems can stop words create in the NLP and why it is 
important to remove them?

Your answer goes here:
'''

'''Exercise 2.2 (2 Points)
Now, write a function that removes the stopwords from the variable counts_ted_top1000 and save it as 
counts_ted_top1000_no_stopword. Use the code for visualization and spot the differences.

The structure in the end should look like this: 
counts_ted_top1000_no_stopword = [(WordA,FrequencyA),(WordB,FrequencyB)]'''

# Your code goes here

mostfreqn = 30  # Here we define how many of them we want to see in the diagramm
frequency = [y for (x, y) in counts_ted_top1000_no_stopword][:mostfreqn]
word = [x for (x, y) in counts_ted_top1000_no_stopword][:mostfreqn]
indices = np.arange(len(counts_ted_top1000_no_stopword[:mostfreqn]))
plt.bar(indices, frequency, color='r')
plt.xticks(indices, word, rotation='vertical')
plt.tight_layout()
plt.show()

'''Wordcloud Visualization
The below so-called wordcloud shows the most frequent words in a larger font and the less frequent ones in a smaller 
font size. It's a quick and cool way of visualizing the most frequent words!'''

# Create a dictionary that maps words to their frequencies
counts_ted_top1000_no_stopword = {word: count for word, count in counts_ted_top1000_no_stopword}

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white')

# Generate the word cloud
wordcloud.generate_from_frequencies(counts_ted_top1000_no_stopword)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

'''Part 3: Generating the Word Embeddings with Word2Vec
Now it is time to run the embedding model. Gensim has an already implemented model that you can use. 
Using the provided model is enough for the purposes of our notebook. If you want to dive deeper into the topic - 
this youtube video https://www.youtube.com/watch?v=kKDYtZfriI8 could be a great guidance for you to get started.'''

# This takes a moment.. dont worry :D
from gensim.models import Word2Vec

model_ted = Word2Vec(sentences_ted)

'''Part 4: Inspection of our learned representations/embeddings (3 Points)
Now that we have a model that captures the word embeddings, we can use it to explore properties of the words in the text.'''

'''First, code a line that looks at the embedding of one individual word/token. What does the representation of "house" 
look like in the embedding model? You may refer to the following gensim docs for functions, that might help you 
https://radimrehurek.com/gensim/models/keyedvectors.html). This will give you 1 point.
'''

# Your solution goes here.

'''The next task for you is to output the most similar word to "town"? This will also give you 1 point.'''

# Your solution goes here.

'''Finally, we want to find out how similar the words "town" and "house" are. Again: 1 point for this!'''

# Your solution goes here.

'''
Exercise 4.1 (3 Points)
Now that we have generated our embeddings, let's test some classical ideas: implement the following formula. Print out the 10 words, that are most similar to this formula:
King-Man+Woman=???
There are two ways of computing similarity in word embeddings:

https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar.html
https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar_cosmul.html 
You should try out both! In this case one of them is better, but both of them are valid methods for computing similarity 
in the word-space.
'''

# Your implementation goes here.

'''
Exercise 4.2 (2 Points)

The expected outcome (Queen) should be one of the top ten most similar words. But there are also a lot of words, 
that you would not expect. Think about where how these words might be connected to the formula. 
Take your time and understand why some of the words (luther, mary, dr, president) might be in this list.

Your answer goes here:

'''

'''
t-SNE visualization
We will use the t-SNE algorithm, given below, for visualization. The so-called t-Distributed Stochastic Neighbor 
Embedding (t-SNE) is an unsupervised and non-linear machine learning technique. It is commonly used for visualizing 
high dimensional data (just like our high dimensional vectors). You do not have to understand the code, its purpose is 
simply to give you an idea of how the data is arranged in high dimensional space.
'''
'''Exercise 4.3 (2 Points)
To use the t-SNE code below, first put a list of the top 50 words (as strings, without stopwords) into 
a variable words_top_ted.'''

# Your implementation goes here.

'''
The following code gets the corresponding vectors from the model, assuming it's called model_ted:
'''
# This assumes words_top_ted is a list of strings, the top 250 words
words_top_vec_ted = model_ted.wv[words_top_ted]

'''
The next few lines are for the t-SNE visualization.
'''

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(words_top_vec_ted)
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:, 0],
                                    x2=words_top_ted_tsne[:, 1],
                                    names=words_top_ted))

p.scatter(x="x1", y="x2", size=8, source=source)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(labels)

show(p)

'''
That's it. We hope you had fun and learned something in the process :-)'''
