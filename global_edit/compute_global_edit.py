import torch
import torch.nn as nn

import clip


def compute_delta_text(target: str, neutral: str, clip: nn.Module):
    """ Computes delta text
    https://github.com/orpatashnik/StyleCLIP/blob/f87a47f7fc39e19a1f6c88293875692b303bf168/global_directions/MapTS.py#L193
    Returns:
        difference
    """

    target_embedding = clip.get_text_embedding(target)
    neutral_embedding = clip.get_text_embedding(neutral)

    delta_t = target_embedding - neutral_embedding
    difference = difference / torch.linalg.norm(difference)
    return difference


def get_relevance(delta_image: torch.Tensor, delta_text: torch.Tensor, threshold: float):
    """
    Gets relevance via inner product

    Args:
        delta_image (C, D): Image embedding delta.
            C - # of Generator Channels
            D - # of Clip Channels
        delta_text (D)
    Returns:
        relevance (C): Relevance for each channel
    """
    # Compute relevance via inner product projections
    rel = delta_image @ delta_text  # C

    # Ignore relevances below certain threhold
    ignore = torch.abs(rel) < threshold  # C
    rel[ignore] = 0

    # Normalize as max is 1
    max_ = torch.abs(rel).max()
    rel /= max_
    return rel


def relevance_to_global_edit(rel):
    raise NotImplementedError


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    breakpoint()


if __name__ == "__main__":
    main()
