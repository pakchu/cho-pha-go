# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Features used by AlphaGo Zero, in approximate order of importance.
Feature                 # Notes
Stone History           16 The stones of each color during the last 8 moves.
Ones                    1  Constant plane of 1s
All features with 8 planes are 1-hot encoded, with plane i marked with 1
only if the feature was equal to i. Any features >= 8 would be marked as 8.

This file includes the features from the first paper as DEFAULT_FEATURES
and the features from AGZ as AGZ_FEATURES.
"""

import numpy as np
import go_tf as go
import utils
import torch

# Resolution/truncation limit for one-hot features
P = 8


def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.uint8)
    capped = np.minimum(feature, planes)
    onehot_index_offsets = np.arange(0, utils.product(
        onehot_features.shape), planes) + capped.ravel()
    # A 0 is encoded as [0,0,0,0], not [1,0,0,0], so we'll
    # filter out any offsets that are a multiple of $planes
    # A 1 is encoded as [1,0,0,0], not [0,1,0,0], so subtract 1 from offsets
    nonzero_elements = (capped != 0).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features


def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco


# TODO(tommadams): add a generic stone_features for all N <= 8
@planes(16)
def stone_features(position: go.Position):
    # a bit easier to calculate it with axis 0 being the 16 board states,
    # and then roll axis 0 to the end.
    features = np.zeros([16, position.N, position.N], dtype=np.uint8)

    num_deltas_avail = position.board_deltas.shape[0]
    cumulative_deltas = np.cumsum(position.board_deltas, axis=0)
    last_eight = np.tile(position.board, [8, 1, 1])
    # apply deltas to compute previous board states
    last_eight[1:num_deltas_avail + 1] -= cumulative_deltas
    # if no more deltas are available, just repeat oldest board.
    last_eight[num_deltas_avail +
               1:] = last_eight[num_deltas_avail].reshape(1, position.N, position.N)

    features[::2] = last_eight == position.to_play
    features[1::2] = last_eight == -position.to_play
    return np.rollaxis(features, 0, 3)


# TODO(tommadams): add a generic stone_features for all N <= 8
@planes(8)
def stone_features_4(position: go.Position):
    # a bit easier to calculate it with axis 0 being the 16 board states,
    # and then roll axis 0 to the end.
    features = np.zeros([8, position.N, position.N], dtype=np.uint8)

    num_deltas_avail = position.board_deltas.shape[0]
    cumulative_deltas = np.cumsum(position.board_deltas, axis=0)
    last = np.tile(position.board, [4, 1, 1])
    # apply deltas to compute previous board states
    last[1:num_deltas_avail + 1] -= cumulative_deltas
    # if no more deltas are available, just repeat oldest board.
    last[num_deltas_avail + 1:] = last[num_deltas_avail].reshape(1, position.N, position.N)

    features[::2] = last == position.to_play
    features[1::2] = last == -position.to_play
    return np.rollaxis(features, 0, 3)


@planes(1)
def color_to_play_feature(position: go.Position):
    if position.to_play == 1:
        return np.ones([position.N, position.N, 1], dtype=np.uint8)
    else:
        return np.zeros([position.N, position.N, 1], dtype=np.uint8)


@planes(3)
def stone_color_feature(position: go.Position):
    board = position.board
    features = np.zeros([position.N, position.N, 3], dtype=np.uint8)
    if position.to_play == 1:
        features[board == 1, 0] = 1
        features[board == -1, 1] = 1
    else:
        features[board == -1, 0] = 1
        features[board == 1, 1] = 1

    features[board == 0, 2] = 1
    return features


@planes(1)
def ones_feature(position: go.Position):
    return np.ones([position.N, position.N, 1], dtype=np.uint8)


@planes(P)
def recent_move_feature(position: go.Position):
    onehot_features = np.zeros([position.N, position.N, P], dtype=np.uint8)
    for i, player_move in enumerate(reversed(position.recent[-P:])):
        _, move = player_move  # unpack the info from position.recent
        if move is not None:
            onehot_features[move[0], move[1], i] = 1
    return onehot_features


@planes(P)
def liberty_feature(position: go.Position):
    return make_onehot(position.get_liberties(), P)


@planes(3)
def few_liberties_feature(position: go.Position):
    feature = position.get_liberties()
    onehot_features = np.zeros(feature.shape + (3,), dtype=np.uint8)
    onehot_index_offsets = np.arange(0, utils.product(
        onehot_features.shape), 3) + feature.ravel()
    nonzero_elements = ((feature != 0) & (feature <= 3)).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features


@planes(1)
def would_capture_feature(position: go.Position):
    features = np.zeros([position.N, position.N, 1], dtype=np.uint8)
    for g in position.lib_tracker.groups.values():
        if g.color == position.to_play:
            continue
        if len(g.liberties) == 1:
            lib = next(iter(g.liberties))
            features[lib + (0,)] = 1
    return features


# DEFAULT_FEATURES = [
#     stone_color_feature,
#     ones_feature,
#     liberty_feature,
#     recent_move_feature,
#     would_capture_feature,
# ]

# DEFAULT_FEATURES_PLANES = sum(f.planes for f in DEFAULT_FEATURES)

# AGZ_FEATURES = [
#     stone_features,
#     color_to_play_feature
# ]

# AGZ_FEATURES_PLANES = sum(f.planes for f in AGZ_FEATURES)

# MLPERF07_FEATURES = [
#     stone_features_4,
#     color_to_play_feature,
#     few_liberties_feature,
#     would_capture_feature,
# ]

# MLPERF07_FEATURES_PLANES = sum(f.planes for f in MLPERF07_FEATURES)


def extract_features(position, features):
    return np.concatenate([feature(position) for feature in features], axis=2)

def position_to_tensor(pos: go.Position) -> torch.Tensor:
    default_features = [
        stone_color_feature,
        ones_feature,
        liberty_feature,
        recent_move_feature,
        would_capture_feature,
    ]
    features = extract_features(pos, default_features)
    # return [feature space, board size, board size]
    return torch.tensor(features, dtype=torch.float32).permute(2, 0, 1)
