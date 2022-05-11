import re
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import glob

file_names = []
#corpus_words = []
#distinct_words = []
#word_idx_dict = {}

#--------------------------------
import random

def weighted_choice(objects, weights):
	""" returns randomly an element from the sequence of 'objects', 
		the likelihood of the objects is weighted according 
		to the sequence of 'weights', i.e. percentages."""

	weights = np.array(weights, dtype=np.float64)
	sum_of_weights = weights.sum()
	# standardization:
	np.multiply(weights, 1 / sum_of_weights, weights)
	weights = weights.cumsum()
	x = random.random()
	for i in range(len(weights)):
		if x < weights[i]:
			return objects[i]
			
#--------------------------------
# set up the sparse matrix

def get_word_set(corpus_words, distinct_words, word_idx_dict, k):
	sets_of_k_words = [ ' '.join(corpus_words[i:i+k]) for i, _ in enumerate(corpus_words[:-k]) ]

	from scipy.sparse import dok_matrix

	sets_count = len(list(set(sets_of_k_words)))
	next_after_k_words_matrix = dok_matrix((sets_count, len(distinct_words)))

	distinct_sets_of_k_words = list(set(sets_of_k_words))
	k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}

	for i, word in enumerate(sets_of_k_words[:-k]):

		word_sequence_idx = k_words_idx_dict[word]
		next_word_idx = word_idx_dict[corpus_words[i+k]]
		next_after_k_words_matrix[word_sequence_idx, next_word_idx] +=1
		
	return (distinct_sets_of_k_words, next_after_k_words_matrix, k_words_idx_dict)

#--------------------------------
# sampling

def sample_next_word_after_sequence(sets, distinct_words, word_sequence, seed_length, alpha = 0):

	next_after_k_words_matrix = sets[seed_length-1][1]
	k_words_idx_dict = sets[seed_length-1][2]
	
	try:
		next_after_k_words_matrix[k_words_idx_dict[word_sequence]]
	except:
		if seed_length == 1:
			return None
		else:
			# try the next shortest sequence
			word_sequence = word_sequence[1:]
			return sample_next_word_after_sequence(word_sequence, distinct_words, seed_length-1, alpha)

	next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
	likelihoods = next_word_vector/next_word_vector.sum()
	
	return weighted_choice(distinct_words, likelihoods.toarray())

#--------------------------------
def stochastic_chain(sets, distinct_words, seed, chain_length=15, seed_length=3, alpha=0):
	current_words = seed.split(' ')
	if len(current_words) != seed_length:
		raise ValueError(f'wrong number of words, expected {seed_length}')
	sentence = seed

	for _ in range(chain_length):
		sentence+=' '
		next_word = sample_next_word_after_sequence(sets, distinct_words, ' '.join(current_words), seed_length, alpha)
		
		if next_word is None:
			return sentence
		else:
			sentence+=next_word
			current_words = current_words[1:]+[next_word]
		
	return sentence
	
#--------------------------------
def get_corpus(file_names):
	corpus = ""
	for file_name in file_names:
		with open(file_name, 'r') as f:
			for s in f.readlines():
				if not s.startswith('#') and not s.startswith('=='):
					corpus += s

	corpus = corpus.lower()

	corpus = corpus.replace('\n',' %NEWLINE% ')
	corpus = corpus.replace('\t',' ')
	corpus = corpus.replace(',', ' ')
	corpus = corpus.replace('.', ' ')
	corpus = corpus.replace('(', ' ')
	corpus = corpus.replace(')', ' ')
	corpus = corpus.replace('?', ' ')
	corpus = corpus.replace('!', ' ')
	corpus = corpus.replace('=====================================================', ' ')

	print("corpus: %i characters" % len(corpus))
	
	return corpus

#--------------------------------
def examine_corpus(corpus):
	corpus_words = corpus.split(' ')
	corpus_words = [word for word in corpus_words if word != '']
	print("%i words" % len(corpus_words))

	distinct_words = list(set(corpus_words))
	word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
	distinct_words_count = len(list(set(corpus_words)))
	print("%i distinct words" % distinct_words_count)
	
	return (corpus_words, distinct_words, word_idx_dict)
	
#--------------------------------
def generate(corpus_words, distinct_words, word_idx_dict, max_k, start=None):
	sets = [get_word_set(corpus_words, distinct_words, word_idx_dict, 1), 
		get_word_set(corpus_words, distinct_words, word_idx_dict, 2), 
		get_word_set(corpus_words, distinct_words, word_idx_dict, 3)]

	# generate	
	if start is None:
		seed = random.choice(sets[max_k-1][0])
	else:
		seed = start
		
	chain_len = 20 + int(random.gauss(20,8))
	s = stochastic_chain(sets, distinct_words, seed, chain_len, max_k, 0)
	
	#s = s.replace('%NEWLINE%', '\n')

	#print('=====================================================')
	#print(s)
	#print('=====================================================')
	
	return s

#--------------------------------
# Read the corpus

if len(sys.argv) == 1:
	print("USAGE: python rvriffs.py <corpus files>")
	print("Options:")
	print(" -start-note:XXX : specify the note to start with")
	exit(1)
	
start_note = None

for arg in sys.argv[1:]:
	if "-start-note:" in arg:
		start_note = arg[12:]
	else:
		file_names.append(arg)

corpus = get_corpus(file_names)

# split into times and notes
# eg: 8dB4 16C5 4C5 2C#5

time_corpus = ""
note_corpus = ""

for word in corpus.split(' '):
	if "%NEWLINE%" not in word:
		if "#" in word:
			time = word[:-3]
			note = word[-3:]
		elif "r" in word:
			time = word[:-1]
			note = word[-1:]
		else:
			time = word[:-2]
			note = word[-2:]
		
		time_corpus += time+" "
		note_corpus += note+" "
	else:
		#time_corpus += "%NEWLINE%" ignore because we want new lines to trigger on notes only
		note_corpus += "%NEWLINE%"
		pass

# generate random times and notes

(corpus_words, distinct_words, word_idx_dict) = examine_corpus(time_corpus)
gen_times = generate(corpus_words, distinct_words, word_idx_dict, max_k = 1)

(corpus_words, distinct_words, word_idx_dict) = examine_corpus(note_corpus)
gen_notes = generate(corpus_words, distinct_words, word_idx_dict, max_k = 1, start = start_note)

# stitch them together
result = []

times = gen_times.split(' ')
i = 0
for note in gen_notes.split(' '):
	if "%NEWLINE%" in note:
		result.append('\n')
	elif len(times)-1 > i:
		i += 1
		result.append(times[i]+note)

print('=====================================================')
print(" ".join(result))
print('=====================================================')
	
	
