# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance generation from score in Tensor2Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import glob
import itertools
import os
import random
import sys

import concurrent
from tqdm import tqdm

import magenta
from magenta.models.score2perf import modalities
from magenta.models.score2perf import music_encoders
from magenta.models.score2perf.datagen_beam import SCORE_BPM, DataAugmentationError
from magenta.music import chord_symbols_lib, chord_inference, melody_inference
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
from tensor2tensor.data_generators import problem, generator_utils
from tensor2tensor.layers import modalities as t2t_modalities
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
import tensorflow as tf
from magenta.models.score2perf import datagen_beam

# TODO(iansimon): figure out the best way not to hard-code these constants

NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108

# pylint: disable=line-too-long
MAESTRO_TFRECORD_PATHS = {
    'train': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_train.tfrecord',
    'dev': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_validation.tfrecord',
    'test': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_test.tfrecord'
}
# pylint: enable=line-too-long


class Score2PerfProblem(problem.Problem):
  """Base class for musical score-to-performance problems.

  Data files contain tf.Example protos with encoded performance in 'targets' and
  optional encoded score in 'inputs'.
  """

  @property
  def splits(self):
    """Dictionary of split names and probabilities. Must sum to one."""
    raise NotImplementedError()

  @property
  def min_hop_size_seconds(self):
    """Minimum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def max_hop_size_seconds(self):
    """Maximum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def num_replications(self):
    """Number of times entire input performances will be split."""
    return 1

  @property
  def add_eos_symbol(self):
    """Whether to append EOS to encoded performances."""
    raise NotImplementedError()

  @property
  def absolute_timing(self):
    """Whether or not score should use absolute (vs. tempo-relative) timing."""
    return False

  @property
  def stretch_factors(self):
    """Temporal stretch factors for data augmentation (in datagen)."""
    return [1.0]

  @property
  def transpose_amounts(self):
    """Pitch transposition amounts for data augmentation (in datagen)."""
    return [0]

  @property
  def random_crop_length_in_datagen(self):
    """Randomly crop targets to this length in datagen."""
    return None

  @property
  def random_crop_in_train(self):
    """Whether to randomly crop each training example when preprocessing."""
    return False

  @property
  def split_in_eval(self):
    """Whether to split each eval example when preprocessing."""
    return False

  def performances_input_transform(self, tmp_dir):
    """Input performances beam transform (or dictionary thereof) for datagen."""
    raise NotImplementedError()

  @staticmethod
  def process_midi(self, f):

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
        ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
          augmented_ns, transpose_amount,
          min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
          in_place=True)
      except chord_symbols_lib.ChordSymbolError:
        raise datagen_beam.DataAugmentationError(
          'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationError(
          'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    self._min_hop_size_seconds = 0.0
    self._max_hop_size_seconds = 0.0
    self._num_replications = 1
    self._encode_performance_fn = self.performance_encoder().encode_note_sequence
    self._encode_score_fns = dict((name, encoder.encode_note_sequence)
                                  for name, encoder in self.score_encoders())

    augment_params = itertools.product(
      self.stretch_factors, self.transpose_amounts)
    augment_fns = [
      functools.partial(augment_note_sequence,
                        stretch_factor=s, transpose_amount=t)
      for s, t in augment_params
    ]

    self._augment_fns = augment_fns
    self._absolute_timing = self.absolute_timing
    self._random_crop_length = self.random_crop_length_in_datagen
    if self._random_crop_length is not None:
      self._augment_fns = self._augment_fns

    rets = []
    ns = magenta.music.midi_file_to_sequence_proto(f)
    # Apply sustain pedal.
    ns = sequences_lib.apply_sustain_control_changes(ns)

    # Remove control changes as there are potentially a lot of them and they are
    # no longer needed.
    del ns.control_changes[:]

    if (self._min_hop_size_seconds and
            ns.total_time < self._min_hop_size_seconds):
      print("sequence_too_short")
      return []

    sequences = []
    for _ in range(self._num_replications):
      if self._max_hop_size_seconds:
        if self._max_hop_size_seconds == self._min_hop_size_seconds:
          # Split using fixed hop size.
          sequences += sequences_lib.split_note_sequence(
            ns, self._max_hop_size_seconds)
        else:
          # Sample random hop positions such that each segment size is within
          # the specified range.
          hop_times = [0.0]
          while hop_times[-1] <= ns.total_time - self._min_hop_size_seconds:
            if hop_times[-1] + self._max_hop_size_seconds < ns.total_time:
              # It's important that we get a valid hop size here, since the
              # remainder of the sequence is too long.
              max_offset = min(
                self._max_hop_size_seconds,
                ns.total_time - self._min_hop_size_seconds - hop_times[-1])
            else:
              # It's okay if the next hop time is invalid (in which case we'll
              # just stop).
              max_offset = self._max_hop_size_seconds
            offset = random.uniform(self._min_hop_size_seconds, max_offset)
            hop_times.append(hop_times[-1] + offset)
          # Split at the chosen hop times (ignoring zero and the final invalid
          # time).
          sequences += sequences_lib.split_note_sequence(ns, hop_times[1:-1])
      else:
        sequences += [ns]

    for performance_sequence in sequences:
      if self._encode_score_fns:
        # We need to extract a score.
        if not self._absolute_timing:
          # Beats are required to extract a score with metric timing.
          beats = [
            ta for ta in performance_sequence.text_annotations
            if (ta.annotation_type ==
                music_pb2.NoteSequence.TextAnnotation.BEAT)
               and ta.time <= performance_sequence.total_time
          ]
          if len(beats) < 2:
            print('not_enough_beats')
            continue

          # Ensure the sequence starts and ends on a beat.
          performance_sequence = sequences_lib.extract_subsequence(
            performance_sequence,
            start_time=min(beat.time for beat in beats),
            end_time=max(beat.time for beat in beats)
          )

          # Infer beat-aligned chords (only for relative timing).
          try:
            chord_inference.infer_chords_for_sequence(
              performance_sequence,
              chord_change_prob=0.25,
              chord_note_concentration=50.0,
              add_key_signatures=True)
          except chord_inference.ChordInferenceError:
            print("chord_inference_failed")
            continue

        # Infer melody regardless of relative/absolute timing.
        try:
          melody_instrument = melody_inference.infer_melody_for_sequence(
            performance_sequence,
            melody_interval_scale=2.0,
            rest_prob=0.1,
            instantaneous_non_max_pitch_prob=1e-15,
            instantaneous_non_empty_rest_prob=0.0,
            instantaneous_missing_pitch_prob=1e-15)
        except melody_inference.MelodyInferenceError:
          print('melody_inference_failed')
          continue

        if not self._absolute_timing:
          # Now rectify detected beats to occur at fixed tempo.
          # TODO(iansimon): also include the alignment
          score_sequence, unused_alignment = sequences_lib.rectify_beats(
            performance_sequence, beats_per_minute=SCORE_BPM)
        else:
          # Score uses same timing as performance.
          score_sequence = copy.deepcopy(performance_sequence)

        # Remove melody notes from performance.
        performance_notes = []
        for note in performance_sequence.notes:
          if note.instrument != melody_instrument:
            performance_notes.append(note)
        del performance_sequence.notes[:]
        performance_sequence.notes.extend(performance_notes)

        # Remove non-melody notes from score.
        score_notes = []
        for note in score_sequence.notes:
          if note.instrument == melody_instrument:
            score_notes.append(note)
        del score_sequence.notes[:]
        score_sequence.notes.extend(score_notes)

        # Remove key signatures and beat/chord annotations from performance.
        del performance_sequence.key_signatures[:]
        del performance_sequence.text_annotations[:]

      for augment_fn in self._augment_fns:
        # Augment and encode the performance.
        try:
          augmented_performance_sequence = augment_fn(performance_sequence)
        except DataAugmentationError as e:
          print("augment_performance_failed", e)
          continue
        example_dict = {
          'targets': self._encode_performance_fn(
            augmented_performance_sequence)
        }
        if not example_dict['targets']:
          print('skipped_empty_targets')
          continue

        if (self._random_crop_length and
                len(example_dict['targets']) > self._random_crop_length):
          # Take a random crop of the encoded performance.
          max_offset = len(example_dict['targets']) - self._random_crop_length
          offset = random.randrange(max_offset + 1)
          example_dict['targets'] = example_dict['targets'][
                                    offset:offset + self._random_crop_length]

        if self._encode_score_fns:
          # Augment the extracted score.
          try:
            augmented_score_sequence = augment_fn(score_sequence)
          except DataAugmentationError:
            print('augment_score_failed')
            continue

          # Apply all score encoding functions.
          skip = False
          for name, encode_score_fn in self._encode_score_fns.items():
            example_dict[name] = encode_score_fn(augmented_score_sequence)
            if not example_dict[name]:
              print('skipped_empty_%s' % name)
              skip = True
              break
          if skip:
            continue

        rets.append(example_dict)
    return rets

  def generator(self, files):
    STEPS = 1000
    for i in range(len(files) // STEPS + 1):
      with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() * 1.5)) as executor:
        future_to_url = {
          executor.submit(self.process_midi, self, f): idx for idx, f in enumerate(files[i * STEPS: i * STEPS + STEPS])
        }

      for future in tqdm(concurrent.futures.as_completed(future_to_url)):
        idx = future_to_url[future]
        data = future.result()
        for d in data:
          yield d

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
      data_dir, 10, shuffled=False)
    dev_paths = self.dev_filepaths(
      data_dir, 1, shuffled=True)

    midi_files = glob.glob('data/maestro/maestro-v2.0.0/*/*.midi')
    random.seed(13)
    random.shuffle(midi_files)

    generator_utils.generate_files(
      self.generator(midi_files[:50]), dev_paths)

    generator_utils.generate_files(
      self.generator(midi_files[50:]), train_paths)
    generator_utils.shuffle_dataset(train_paths)

  # def generate_data(self, data_dir, tmp_dir, task_id=-1):
  #   del task_id
  #
  #   def augment_note_sequence(ns, stretch_factor, transpose_amount):
  #     """Augment a NoteSequence by time stretch and pitch transposition."""
  #     augmented_ns = sequences_lib.stretch_note_sequence(
  #         ns, stretch_factor, in_place=False)
  #     try:
  #       _, num_deleted_notes = sequences_lib.transpose_note_sequence(
  #           augmented_ns, transpose_amount,
  #           min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
  #           in_place=True)
  #     except chord_symbols_lib.ChordSymbolError:
  #       raise datagen_beam.DataAugmentationError(
  #           'Transposition of chord symbol(s) failed.')
  #     if num_deleted_notes:
  #       raise datagen_beam.DataAugmentationError(
  #           'Transposition caused out-of-range pitch(es).')
  #     return augmented_ns
  #
  #   augment_params = itertools.product(
  #       self.stretch_factors, self.transpose_amounts)
  #   augment_fns = [
  #       functools.partial(augment_note_sequence,
  #                         stretch_factor=s, transpose_amount=t)
  #       for s, t in augment_params
  #   ]
  #
  #   datagen_beam.generate_examples(
  #       input_transform=self.performances_input_transform(tmp_dir),
  #       output_dir=data_dir,
  #       problem_name=self.dataset_filename(),
  #       splits=self.splits,
  #       min_hop_size_seconds=self.min_hop_size_seconds,
  #       max_hop_size_seconds=self.max_hop_size_seconds,
  #       min_pitch=MIN_PITCH,
  #       max_pitch=MAX_PITCH,
  #       num_replications=self.num_replications,
  #       encode_performance_fn=self.performance_encoder().encode_note_sequence,
  #       encode_score_fns=dict((name, encoder.encode_note_sequence)
  #                             for name, encoder in self.score_encoders()),
  #       augment_fns=augment_fns,
  #       absolute_timing=self.absolute_timing,
  #       random_crop_length=self.random_crop_length_in_datagen)

  def hparams(self, defaults, model_hparams):
    del model_hparams   # unused
    perf_encoder = self.get_feature_encoders()['targets']
    defaults.modality = {'targets': t2t_modalities.ModalityType.SYMBOL}
    defaults.vocab_size = {'targets': perf_encoder.vocab_size}
    if self.has_inputs:
      score_encoder = self.get_feature_encoders()['inputs']
      if isinstance(score_encoder.vocab_size, list):
        # TODO(trandustin): We default to not applying any transformation; to
        # apply one, pass modalities.bottom to the model's hparams.bottom. In
        # future, refactor the tuple of the "inputs" feature to be part of the
        # features dict itself, i.e., have multiple inputs each with its own
        # modality and vocab size.
        modality_cls = modalities.ModalityType.IDENTITY
      else:
        modality_cls = t2t_modalities.ModalityType.SYMBOL
      defaults.modality['inputs'] = modality_cls
      defaults.vocab_size['inputs'] = score_encoder.vocab_size

  def performance_encoder(self):
    """Encoder for target performances."""
    return music_encoders.MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=self.add_eos_symbol)

  def score_encoders(self):
    """List of (name, encoder) tuples for input score components."""
    return []

  def feature_encoders(self, data_dir):
    del data_dir
    encoders = {
        'targets': self.performance_encoder()
    }
    score_encoders = self.score_encoders()
    if score_encoders:
      if len(score_encoders) > 1:
        # Create a composite score encoder, only used for inference.
        encoders['inputs'] = music_encoders.CompositeScoreEncoder(
            [encoder for _, encoder in score_encoders])
      else:
        # If only one score component, just use its encoder.
        _, encoders['inputs'] = score_encoders[0]
    return encoders

  def example_reading_spec(self):
    data_fields = {
        'targets': tf.VarLenFeature(tf.int64)
    }
    for name, _ in self.score_encoders():
      data_fields[name] = tf.VarLenFeature(tf.int64)

    # We don't actually "decode" anything here; the encodings are simply read as
    # tensors.
    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    if self.has_inputs:
      # Stack encoded score components depthwise as inputs.
      inputs = []
      for name, _ in self.score_encoders():
        inputs.append(tf.expand_dims(example[name], axis=1))
        del example[name]
      example['inputs'] = tf.stack(inputs, axis=2)

    if self.random_crop_in_train and mode == tf.estimator.ModeKeys.TRAIN:
      # Take a random crop of the training example.
      assert not self.has_inputs
      max_offset = tf.maximum(
          tf.shape(example['targets'])[0] - hparams.max_target_seq_length, 0)
      offset = tf.cond(
          max_offset > 0,
          lambda: tf.random_uniform([], maxval=max_offset, dtype=tf.int32),
          lambda: 0
      )
      example['targets'] = (
          example['targets'][offset:offset + hparams.max_target_seq_length])
      return example

    elif self.split_in_eval and mode == tf.estimator.ModeKeys.EVAL:
      # Split the example into non-overlapping segments.
      assert not self.has_inputs
      length = tf.shape(example['targets'])[0]
      extra_length = tf.mod(length, hparams.max_target_seq_length)
      examples = {
          'targets': tf.reshape(
              example['targets'][:length - extra_length],
              [-1, hparams.max_target_seq_length, 1, 1])
      }
      extra_example = {
          'targets': tf.reshape(
              example['targets'][-extra_length:], [1, -1, 1, 1])
      }
      dataset = tf.data.Dataset.from_tensor_slices(examples)
      extra_dataset = tf.data.Dataset.from_tensor_slices(extra_example)
      return dataset.concatenate(extra_dataset)

    else:
      # If not cropping or splitting, do standard preprocessing.
      return super(Score2PerfProblem, self).preprocess_example(
          example, mode, hparams)


class Chords2PerfProblem(Score2PerfProblem):
  """Base class for musical chords-to-performance problems."""

  def score_encoders(self):
    return [('chords', music_encoders.TextChordsEncoder(steps_per_quarter=1))]


class Melody2PerfProblem(Score2PerfProblem):
  """Base class for musical melody-to-performance problems."""

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class AbsoluteMelody2PerfProblem(Score2PerfProblem):
  """Base class for musical (absolute-timed) melody-to-performance problems."""

  @property
  def absolute_timing(self):
    return True

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoderAbsolute(
            steps_per_second=10, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class LeadSheet2PerfProblem(Score2PerfProblem):
  """Base class for musical lead-sheet-to-performance problems."""

  def score_encoders(self):
    return [
        ('chords', music_encoders.TextChordsEncoder(steps_per_quarter=4)),
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


@registry.register_problem('score2perf_maestro_language_uncropped_aug')
class Score2PerfMaestroLanguageUncroppedAug(Score2PerfProblem):
  """Piano performance language model on the MAESTRO dataset."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return dict(
        (split_name, datagen_beam.ReadNoteSequencesFromTFRecord(tfrecord_path))
        for split_name, tfrecord_path in MAESTRO_TFRECORD_PATHS.items())

  @property
  def splits(self):
    return None

  @property
  def min_hop_size_seconds(self):
    return 0.0

  @property
  def max_hop_size_seconds(self):
    return 0.0

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]

  @property
  def random_crop_in_train(self):
    return True

  @property
  def split_in_eval(self):
    return True


@registry.register_problem('score2perf_maestro_absmel2perf_5s_to_30s_aug10x')
class Score2PerfMaestroAbsMel2Perf5sTo30sAug10x(AbsoluteMelody2PerfProblem):
  """Generate performances from an absolute-timed melody, with augmentation."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return dict(
        (split_name, datagen_beam.ReadNoteSequencesFromTFRecord(tfrecord_path))
        for split_name, tfrecord_path in MAESTRO_TFRECORD_PATHS.items())

  @property
  def splits(self):
    return None

  @property
  def min_hop_size_seconds(self):
    return 5.0

  @property
  def max_hop_size_seconds(self):
    return 30.0

  @property
  def num_replications(self):
    return 10

  @property
  def add_eos_symbol(self):
    return True

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]


@registry.register_hparams
def score2perf_transformer_base():
  hparams = transformer.transformer_base()
  hparams.bottom['inputs'] = modalities.bottom
  hparams.max_length = 0
  hparams.max_target_seq_length = 2048
  hparams.batch_size = 2048 * 3
  hparams.batch_shuffle_size = 128
  hparams.symbol_modality_num_shards = 1
  # hparams.ffn_layer = "conv_relu_conv"
  # hparams.conv_first_kernel = 9
  return hparams
