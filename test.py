import sys
import math
import os
import logging
from six.moves import xrange

import numpy as np
import tensorflow as tf

import data_utils
import translate

_buckets = [(10, 10), (17, 17), (28, 28), (80, 80)]

if __name__ == '__main__':
  inputFile = sys.argv[1]
  fin = open(inputFile, 'r'); fout = open('test-results.txt', 'w')
  with tf.Session() as sess:
    # Create model and load parameters.
    model = translate.create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab, _ = data_utils.initialize_vocabulary("data/vocab30000.from")
    fr_vocab, _ = data_utils.initialize_vocabulary("data/vocab30000.to")
    _, rev_fr_vocab = data_utils.initialize_vocabulary("data/vocab30000.to")

    numTested = 0
    for line in fin.readlines():
      sentences = line.strip().split('\t')
      perps = []
      for sourceIndex in range(0, 2):
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentences[sourceIndex]), en_vocab)
        target_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentences[sourceIndex+1]), fr_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
          if bucket[0] >= len(token_ids):
            bucket_id = i
            break
        else:
          fout.write("Not valid.\n")
          break

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, target_ids)]}, bucket_id)
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        encoder_size, decoder_size = model.buckets[bucket_id]
        input_feed = {}
        for l in xrange(encoder_size):
          input_feed[model.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
          input_feed[model.decoder_inputs[l].name] = decoder_inputs[l]
          input_feed[model.target_weights[l].name] = target_weights[l]
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = model.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)
        # Output feed: depends on whether we do a backward step or not.
        output_feed = [model.losses[bucket_id]]  # Loss for this batch.
        # Calculate perplexity.
        loss = sess.run(output_feed, input_feed)
        perplexity = math.exp(float(loss[0])) if loss[0] < 300 else float("inf")
        # Get logits.
        '''outputs = []
        for i in range(0, len(output_logits)):
          if i == len(target_ids):
            break
          probs = np.exp(output_logits[i][0])
          outputs.append(probs[target_ids[i]]/sum(probs))
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in target_ids:
          outputs = outputs[:target_ids.index(data_utils.EOS_ID)]
        # Calculate perplexity.
        print(outputs)
        sump = 0.0; count = 0
        for i in range(0, len(outputs)):
          if outputs[i] != 0:
            sump += math.log(outputs[i])
            count += 1'''
        perps.append(perplexity)
      if len(perps) == 2:
        fout.write(str(perps[0])+' '+str(perps[1])+'\n')
      numTested += 1
      if numTested%500 == 0:
        print(str(numTested)+" triples are tested.\n")
  fin.close()
  fout.close()
