"""Check the frozen weights required by a portable inference checkpoint."""

import hashlib


def frozen_encoder_signature(model):
    cached = getattr(model, '_frozen_encoder_signature', None)
    if cached is not None:
        return cached
    encoder = getattr(model, 'vit_encoder', None)
    if encoder is None:
        encoder = model.cnn_encoder
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    signature = digest.hexdigest()
    model._frozen_encoder_signature = signature
    return signature
