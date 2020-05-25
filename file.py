# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:13:04 2020

@author: rahul
"""

from Summarizer.Summarization import bart_summarizer, bert_summarizer, xlnet_summarizer, gpt2_summarizer, T5_summarizer
from KeywordExtraction.Keyword_Extractor import  Synonym_Keywords_Generation
from SpellCorrection.Spell_Correction import SpellCheck2
# lib files from question generation to be imported


data = '''
Traditiomal electrnics are connected with wires to the power supply to switches and sensors. Over the time the amount of sensors and IoT devices is growing so there is no more space in future for all the cables needed for the connections.
MID (molded interconnect devices) opens the possibility to place tracks directly on the surface of plastic parts.
MID must be further developed for our own applications. E.g. printing in plastc housing parts.
We expect that this will be the enabler for much more sensors in our tools. 5 years ago the MID technology was not competitive vs. classical cabling/wiring. With increasing electronics, cabling space gets rare. What technologies and companies exist on the market to create MID? Interesting technologies could be laser activation, galvanization, deformable sheets.
'''
# spell check
#corrected_data = SpellCheck2(data)
#print("Did you mean: " + corrected_data)



# summarization
summary,scores = bart_summarizer(data)
print(summary)
print(scores)

# summary,scores = bert_summarizer(data)
# print(summary)
# print(scores)

# summary,scores = xlnet_summarizer(data)
# print(summary)
#print(scores)

#summary,scores = gpt2_summarizer(data)
#print(summary)
#print(scores)

#summary,scores = T5_summarizer(data)
#print(summary)
#print(scores)

# Keyword Extraction
word_scores, key_words, synonyms = Synonym_Keywords_Generation(data)
print(word_scores)
#key_words = str(key_words)[:-2]
print(key_words)
print(synonyms)

lst = [summary]
lst = lst + key_words
print(lst)


# Question Generation