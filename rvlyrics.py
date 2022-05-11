import sys
import pandas as pd
import seaborn as sns
import numpy as np
import glob

file_names = []
corpus_words = []
distinct_words = []
word_idx_dict = {}

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

def get_word_set(k):
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

def sample_next_word_after_sequence(word_sequence, seed_length, alpha = 0):

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
			return sample_next_word_after_sequence(word_sequence, seed_length-1, alpha)

	next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
	likelihoods = next_word_vector/next_word_vector.sum()
	
	return weighted_choice(distinct_words, likelihoods.toarray())

#--------------------------------
def stochastic_chain(seed, chain_length=15, seed_length=3, alpha=0):
	current_words = seed.split(' ')
	if len(current_words) != seed_length:
		raise ValueError(f'wrong number of words, expected {seed_length}')
	sentence = seed

	for _ in range(chain_length):
		sentence+=' '
		next_word = sample_next_word_after_sequence(' '.join(current_words), seed_length, alpha)
		
		if next_word is None:
			return sentence
		else:
			sentence+=next_word
			current_words = current_words[1:]+[next_word]
		
	return sentence
	
#--------------------------------
# Read the corpus

if len(sys.argv) == 1:
	print("USAGE: python rvlyrics.py <corpus files>")
	print("Options:")
	print(" -start-word:XXX : specify the word to start with")
	exit(1)

seed = None
for arg in sys.argv[1:]:
	if "-start-word:" in arg:
		seed = arg[12:]
	else:
		file_names.append(arg)

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

#--------------------------------
# examine corpus

corpus_words = corpus.split(' ')
corpus_words = [word for word in corpus_words if word != '']
#corpus_words # [...'a', 'wyvern', ',', 'two', 'of', 'the', 'thousand'...]
print("%i words" % len(corpus_words))

distinct_words = list(set(corpus_words))
word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
distinct_words_count = len(list(set(corpus_words)))
print("%i distinct words" % distinct_words_count)

max_k = 1

sets = [get_word_set(1), get_word_set(2), get_word_set(3)]

# generate
if seed is None:
	seed = random.choice(sets[max_k-1][0])
chain_len = 20 + int(random.gauss(99,20))
s = stochastic_chain(seed, chain_len, max_k, 0)

s = s.replace('%NEWLINE%', '\n')

print('=====================================================')
print(s)
print('=====================================================')