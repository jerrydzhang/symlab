from typing import TypedDict

import numpy as np
import torch
from symbolic import Evaluated

from manifest.tokenizer import XValsTokenizer, PAD_ID

from torch.nn.utils.rnn import pad_sequence


class SRBatch(TypedDict):
    data: torch.Tensor
    tokens: torch.Tensor
    num_values: torch.Tensor
    data_mask: torch.Tensor
    token_mask: torch.Tensor
    stats: torch.Tensor


def collate_fn(samples: list[Evaluated], tokenizer: XValsTokenizer) -> SRBatch:
    tokens_list = []
    num_values_list = []
    data_list = []
    stats_list = []

    max_inputs = tokenizer.max_inputs

    for sample in samples:
        tokens, num_values = tokenizer.encode(sample.expression)
        tokens_list.append(torch.tensor(tokens, dtype=torch.long))
        num_values_list.append(torch.tensor(num_values, dtype=torch.float))

        X = sample.X
        y = sample.y

        # Pad X to max_inputs columns (unused variables get zeros)
        n_actual = X.shape[1]
        if n_actual < max_inputs:
            X_padded = np.zeros((X.shape[0], max_inputs))
            X_padded[:, :n_actual] = X
        else:
            X_padded = X

        X_mean = X_padded.mean(axis=0)
        X_std = X_padded.std(axis=0)
        y_mean = y.mean()
        y_std = y.std()

        X_norm = (X_padded - X_mean) / (X_std + 1e-8)
        y_norm = (y - y_mean) / (y_std + 1e-8)
        combined_data = np.column_stack((X_norm, y_norm))
        data_list.append(torch.from_numpy(combined_data).float())

        # Normalize constant values: the num_head predicts normalized constants.
        # Real values are recovered at inference via the stats token.
        stats = np.concatenate([X_mean, X_std, [y_mean, y_std]])
        stats_list.append(torch.from_numpy(stats).float())

    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=PAD_ID)
    padded_num_values = pad_sequence(
        num_values_list, batch_first=True, padding_value=1.0
    )
    token_mask = padded_tokens != PAD_ID

    data_lens = torch.tensor([d.shape[0] for d in data_list])
    max_n = int(data_lens.max().item())
    batch_size = len(data_list)
    # n_features + target
    D = data_list[0].shape[1]
    data_padded = torch.zeros(batch_size, max_n, D)
    for i, d in enumerate(data_list):
        data_padded[i, : d.shape[0], :] = d
    data_mask = torch.arange(max_n)[None, :] < data_lens[:, None]

    stats = torch.stack(stats_list, dim=0)
    return SRBatch(
        data=data_padded,
        tokens=padded_tokens,
        num_values=padded_num_values,
        stats=stats,
        data_mask=data_mask,
        token_mask=token_mask,
    )
