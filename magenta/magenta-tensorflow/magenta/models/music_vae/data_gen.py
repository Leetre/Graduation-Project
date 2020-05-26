import os

import concurrent

from tqdm import tqdm

from magenta.models.music_vae import data_hierarchical, data
from magenta.models.music_vae.configs import CONFIG_MAP
from magenta.models.music_vae.data import NoteSequenceAugmenter
from magenta.models.music_vae.data_utils import generate_files, shuffle_dataset, UNSHUFFLED_SUFFIX
from magenta.music import abc_parser
from magenta.music import midi_io
from magenta.music import musicxml_reader
from magenta.music import note_sequence_io
import tensorflow as tf

from magenta.protobuf import music_pb2
import magenta.music as mm
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory containing files to convert.')
tf.app.flags.DEFINE_string('output_file', None,
                           'Path to output TFRecord file. Will be overwritten '
                           'if it already exists.')
tf.app.flags.DEFINE_bool('recursive', True,
                         'Whether or not to recurse into subdirectories.')
tf.app.flags.DEFINE_string('config', 'hier-multiperf_vel_1bar_med', '')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


def get_midi_files(root_dir, sub_dir, recursive=False):
  """Converts files.

  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.
    writer: A TFRecord writer
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.

  Returns:
    A map from the resulting Futures to the file paths being converted.
  """
  dir_to_convert = os.path.join(root_dir, sub_dir)
  tf.logging.info("collect files in '%s'.", dir_to_convert)
  files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
  recurse_sub_dirs = []
  midi_files = []
  for file_in_dir in files_in_dir:
    full_file_path = os.path.join(dir_to_convert, file_in_dir)
    if (full_file_path.lower().endswith('.mid') or
        full_file_path.lower().endswith('.midi')):
      midi_files.append(full_file_path)
    else:
      if recursive and tf.gfile.IsDirectory(full_file_path):
        recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
      else:
        tf.logging.warning(
          'Unable to find a converter for file %s', full_file_path)

  for recurse_sub_dir in recurse_sub_dirs:
    midi_files += get_midi_files(root_dir, recurse_sub_dir, recursive)

  return midi_files


def convert_midi(root_dir, sub_dir, full_file_path, output_file):
  data_converter = CONFIG_MAP[FLAGS.config].data_converter
  augmenter = CONFIG_MAP[FLAGS.config].note_sequence_augmenter
  ret = []

  try:
    sequence = midi_io.midi_to_sequence_proto(
      tf.gfile.GFile(full_file_path, 'rb').read())
  except midi_io.MIDIConversionError as e:
    tf.logging.warning(
      'Could not parse MIDI file %s. It will be skipped. Error was: %s',
      full_file_path, e)
    return []
  sequence.collection_name = os.path.basename(root_dir)
  sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.id = note_sequence_io.generate_note_sequence_id(
    sequence.filename, sequence.collection_name, 'midi')
  # tf.logging.info('Converted MIDI file %s.', full_file_path)

  for s in (augmenter.get_all(sequence) if augmenter is not None else [sequence]):
    data = data_converter.to_tensors(s)
    for inp, c, l in zip(data.inputs, data.controls, data.lengths):
      s = list(inp.shape)
      inp = inp.reshape(-1).tolist()
      c = c.reshape(-1).tolist()
      if len(c) == 0:
        c = [0]
      if isinstance(l, int):
        l = [l]
      ret.append({
        'notes': inp,
        'chords': c,
        'shape': s,
        'lengths': l
      })
  if len(ret) > 0:
    np.save("{}_npy/{}".format(output_file, os.path.basename(full_file_path)), ret)
  return ret


def generator(root_dir, output_file, recursive=False):
  midi_files = get_midi_files(root_dir, '', recursive)
  STEPS = 10000
  seg_idx = 0
  
  # os.makedirs('{}_npy'.format(output_file), exist_ok=True)
  # for i in range(len(midi_files) // STEPS + 1):
  #   t = tqdm(midi_files[i * STEPS:(i + 1) * STEPS], total=len(midi_files), initial=STEPS * i, ncols=100)
  #   for full_file_path in t:
  #     for r in convert_midi(root_dir, '', full_file_path, output_file):
  #       r['id'] = [seg_idx]
  #       yield r
  #       seg_idx += 1
  #       t.set_description("total: {}".format(seg_idx))

  os.makedirs('{}_npy'.format(output_file), exist_ok=True)
  for i in range(len(midi_files) // STEPS + 1):
    print(i)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() * 1.5)) as executor:
      # executor.map(convert_midi, [(root_dir, '', full_file_path) for full_file_path in
      #                             midi_files[i * STEPS:(i + 1) * STEPS]])
      futures = [executor.submit(convert_midi, root_dir, '', full_file_path, output_file) for full_file_path in
                 midi_files[i * STEPS:(i + 1) * STEPS]]
      t = tqdm(concurrent.futures.as_completed(futures), total=len(midi_files), initial=STEPS * i, ncols=100)
      for future in t:
        for r in future.result():
          r['id'] = [seg_idx]
          yield r
          seg_idx += 1
        t.set_description("total: {}".format(seg_idx))

#   python magenta/models/music_vae/data_gen.py --input_dir=data/lmd/lmd_full --output_file=data/lmd/lmd_full2 --recursive --config=hier-multiperf_vel_1bar_med

#  python magenta/models/music_vae/data_gen.py --input_dir=data/maestro/maestro-v2.0.0 --output_file=data/maestro/maestro --recursive
def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not FLAGS.input_dir:
    tf.logging.fatal('--input_dir required')
    return
  if not FLAGS.output_file:
    tf.logging.fatal('--output_file required')
    return

  input_dir = os.path.expanduser(FLAGS.input_dir)
  output_file = os.path.expanduser(FLAGS.output_file)
  output_dir = os.path.dirname(output_file)

  if output_dir:
    tf.gfile.MakeDirs(output_dir)

  OUTPUT_SHARDS = 10
  output_files = ["{}_{}.tfrecord{}".format(output_file, f, UNSHUFFLED_SUFFIX) for f in range(OUTPUT_SHARDS)]
  generate_files(generator(input_dir, output_file, FLAGS.recursive), output_files)
  shuffle_dataset(output_files)


def console_entry_point():
  tf.app.run(main) 

if __name__ == '__main__':
  console_entry_point()