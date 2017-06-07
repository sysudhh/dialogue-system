import sys
import math
import os
import logging

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

    # Calculate perplexity.
    for line in fin.readlines():
      sentences = line.strip().split('\t')
      perps = []
      for i in range(0, 2):
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentences[i]), en_vocab)
        target_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentences[i+1]), fr_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
          if bucket[0] >= len(token_ids):
            bucket_id = i
            break
        else:
          logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        # Get logits.
        outputs = []
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
            count += 1
        perps.append(pow(2, -(sump/count)))
      fout.write(str(perps[0])+' '+str(perps[1])+'\n')
  fin.close()
  fout.close()
