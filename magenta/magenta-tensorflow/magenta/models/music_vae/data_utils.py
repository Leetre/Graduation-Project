import random

import six
import tensorflow as tf
import numpy as np

UNSHUFFLED_SUFFIX = "-unshuffled"


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s" % str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


def outputs_exist(filenames):
  for out_fname in filenames:
    out_fname = out_fname.replace(UNSHUFFLED_SUFFIX, "")
    if tf.gfile.Exists(out_fname):
      return out_fname


def generate_files(generator, output_filenames,
                   max_cases=None, cycle_every_n=1):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
    cycle_every_n: how many cases from the generator to take before
      switching to the next shard; by default set to 1, switch every case.
  """
  if outputs_exist(output_filenames):
    tf.logging.info("Skipping generator because outputs files exists at {}"
                    .format(output_filenames))
    return
  tmp_filenames = [fname + ".incomplete" for fname in output_filenames]
  num_shards = len(output_filenames)
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filenames]
  counter, shard = 0, 0
  for case in generator:
    if case is None:
      continue
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    example = to_example(case)
    writers[shard].write(example.SerializeToString())
    if counter % cycle_every_n == 0:
      shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filenames, output_filenames):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info("Generated %s Examples", counter)


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("read: %d", len(records))
  return records


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count > 0 and count % 100000 == 0:
      tf.logging.info("write: %d", count)
  writer.close()


def _shuffle_single(fname):
  records = read_records(fname)
  random.shuffle(records)
  out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
  write_records(records, out_fname)
  tf.gfile.Remove(fname)


def shuffle_dataset(filenames):
  """Shuffles the dataset."""
  if outputs_exist(filenames):
    tf.logging.info("Skipping shuffle because output files exist")
    return
  tf.logging.info("Shuffling data...")
  for filename in filenames:
    _shuffle_single(filename)
  tf.logging.info("Data shuffled.")


def decode_example(data_fields):
  def _decode_example(serialized_example):
    data_items_to_decoders = {
      field: tf.contrib.slim.tfexample_decoder.Tensor(field)
      for field in data_fields
    }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
      data_fields, data_items_to_decoders)

    decode_items = list(sorted(data_items_to_decoders))
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))

  return _decode_example
