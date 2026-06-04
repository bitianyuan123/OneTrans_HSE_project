from typing import Tuple

import torch


def create_masked_tensor(
    data_tensor: torch.Tensor, lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a batch of flattened variable-length sequences into a padded tensor and mask.
    Supports:
      - indices: data shape (total_num_elements,)
      - embeddings/features: data shape (total_num_elements, d1, d2, ...)

    Parameters
    ----------
    data_tensor : torch.Tensor
        Input tensor containing flattened sequences:
        - For indices: shape (total_num_elements,)
        - For embeddings: shape (total_num_elements, embedding_dim)
    lengths : torch.Tensor
        1D tensor of sequence lengths, shape (batch_size,). Specifies the actual length
        of each sequence.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - padded_tensor: Padded tensor of shape:
            - (batch_size, max_seq_len) for indices
            - (batch_size, max_seq_len, embedding_dim) for embeddings
            Shorter sequences are right-padded with zeros.
        - mask: Boolean mask of shape (batch_size, max_seq_len) where True indicates
            valid elements and False indicates padding. Can be used in attention or loss computation.

    Examples
    --------
    >>> data_tensor = torch.tensor([1, 2, 3, 4, 5, 6])  # sequences: [1,2], [3,4,5], [6]
    >>> lengths = torch.tensor([2, 3, 1])
    >>> padded, mask = create_masked_tensor(data_tensor, lengths)
    >>> padded
    tensor([[1, 2, 0],
            [3, 4, 5],
            [6, 0, 0]])
    >>> mask
    tensor([[ True,  True, False],
            [ True,  True,  True],
            [ True, False, False]])
    """
    max_seq_length = lengths.max().cpu().item()
    batch_size = len(lengths)

    if data_tensor.dim() == 1:
        output_shape = (batch_size, max_seq_length)
    else:
        element_shape = data_tensor.shape[1:]
        output_shape = (batch_size, max_seq_length, *element_shape)

    padded_tensor = torch.zeros(output_shape, dtype=data_tensor.dtype, device=data_tensor.device)
    padding_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=data_tensor.device)

    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + length.cpu().item()
        seq = data_tensor[start_idx:end_idx]
        padded_tensor[i, :length] = seq
        padding_mask[i, :length] = True
        start_idx = end_idx

    return padded_tensor, padding_mask
