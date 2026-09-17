from functools import reduce
from einops import rearrange, pack, unpack

import torch
from torch import nn

from torchaudio.functional import resample

from vector_quantize_pytorch import ResidualVQ

from rvqwhisper import RVQFasterWhisperWrapper
from bigvgan import BigVGANWrapper

from encodec.utils import _linear_overlap_add

import numpy as np

# helper functions

def exists(val):
    return val is not None

# hacky way to get num quantizers

def get_num_quantizers(WW: RVQFasterWhisperWrapper):
    return WW.config['quantization']['num_quantizers']

class WhisperBigVGANWrapper():
    """
    Whisper and BigVGAN pipeline as a speech codec.
    TODO:
    """
    def __init__(
        self,
        WhisperWrapper: RVQFasterWhisperWrapper,
        BVGWrapper: BigVGANWrapper,
        speaker,
        num_quantizers = 8,
        device= "cuda"
    ):
        super().__init__()
        # Instantiate a pretrained EnCodec model
        self.WW = WhisperWrapper
        self.BW = BVGWrapper
        self.speaker = speaker 

        num_quantizers = get_num_quantizers(self.WW)

        # Fields that SoundStream has that get used externally. We replicate them here.

        self.codebook_dim = self.WW.config['quantization']['dim']
        self.rq_groups = 1
        self.num_quantizers = num_quantizers

        # cross entropy loss to indices passed in on l2 distance logits introduced in vector-quantize-pytorch 1.2.2

        self.rq = ResidualVQ(
            dim = self.codebook_dim,
            codebook_size = self.WW.config['quantization']['codebook_size'],
            num_quantizers = num_quantizers
        ).to(device)

        # copy codebook over to ResidualVQ for cross entropy loss logic from naturalspeech2

        # for ww_rq_layer, rq_layer in zip(self.WW.model.model.model.vector_quantization.layers, self.rq.layers):
        for ww_rq_layer, rq_layer in zip(self.BW.rvq_model.layers, self.rq.layers):
            ww_codebook = dict(ww_rq_layer._codebook.named_buffers()).get('embed')
            vq_codebook = dict(rq_layer._codebook.named_buffers()).get('embed')

            # ww_codebook = rearrange(ww_codebook, '... -> 1 ...')
            vq_codebook.copy_(ww_codebook)

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def forward(
        self,
        x,
        input_sample_hz = None,
        return_encoded = False,
        **kwargs
    ):

        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        # kwargs for stuff like return_encoded=True, which SoundStream uses but Encodec doesn't
        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."
        # Unlike in the Encodec sample code in its README, x has already been resampled so we don't need to call
        # convert_audio and unsqueeze. The convert_audio function also doesn't play nicely with batches.

        # b = batch, t = timesteps, 1 channel for the 24kHz model, 2 channels for the 48kHz model
        wav = rearrange(x, f'b t -> b {self.model.channels} t')

        # Extract discrete codes from EnCodec
        with torch.inference_mode():
            encoded_frames = self.model.encode(wav)
        # encoded_frames is a list of (frame, scale) tuples. Scale is a scalar but we don't use it. Frame is a tensor
        # of shape [batch, num_quantizers, num_samples_per_frame]. We want to concatenate the frames to get all the
        # timesteps concatenated.
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [batch, num_quantizers, timesteps]
        # transformer code that uses codec expects codes to be [batch, timesteps, num_quantizers]
        codes = rearrange(codes, 'b q n -> b n q')  # result: [batch, timesteps, num_quantizers]
        # in original soundstream, is x, indices, commit_loss. But we only use indices in eval mode, so just keep that.

        # allow for returning of sum of quantized embeddings

        emb = None

        if return_encoded:
            emb = self.get_emb_from_indices(codes)
            emb, = unpack(emb, ps, '* n c')

        codes, = unpack(codes, ps, '* n q')

        return emb, codes, None

    def decode_from_codebook_indices(self, quantized_indices, f0_tokens):
        # Input: batch x num tokens x num quantizers
        # Output: batch x 1 x num samples

        # The following code is hacked in from self.model.decode() (Encodec version 0.1.1) where we skip the part about
        # scaling.
        # Shape: 1 x (num_frames * stride product). 1 because we have 1 frame (because no segmenting)
        frames = self._decode_frame(quantized_indices, f0_tokens)
        return frames
        # result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        # TODO: I'm not overly pleased with this because when this function gets called, we just rearrange the result
        #   back to b n anyways, but we'll keep this as a temporary hack just to make things work for now
        # return rearrange(result, 'b n -> b 1 n')

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')

    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)

    def _decode_frame(self, quantized_indices, f0_tokens):
        # The following code is hacked in from self.model._decode_frame() (Encodec version 0.1.1) where we assume we've
        # already unwrapped the EncodedFrame
        # Input: batch x num tokens x num quantizers
        # Output: batch x new_num_samples, where new_num_samples is num_frames * stride product (may be slightly
        # larger than original num samples as a result, because the last frame might not be "fully filled" with samples
        # if num_samples doesn't divide perfectly).
        # num_frames == the number of acoustic tokens you have, one token per frame
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        

        emb = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            layer = self.rq.layers[i]
            quantized = layer.get_codes_from_indices(indices)
            emb = emb + quantized
    
        # emb = self.rq.decode(codes)
        # emb shape: batch x self.model.quantizer.dimension x T. Note self.model.quantizer.dimension is the embedding dimension
        # breakpoint()
        wavs = []
        for ind in range(f0_tokens.shape[0]):
            new_pitch = self.speaker.convert_pitch(f0_tokens[f0_tokens >= 0].detach().cpu().numpy())
            wav = self.BW.synthesize(emb[ind].detach().cpu().numpy(), self.speaker.xvector, new_pitch, chunk_size = 100)
            # wav = self.BW.synthesize(emb[ind].detach().cpu().numpy(), self.xvector, np.asarray([x for x in f0_tokens[ind].detach().cpu().numpy() if x >= 0]), chunk_size = 50)
            wavs.append(wav[0])
        max_wav_len = max(len(arr) for arr in wavs)
        stacked = np.array([np.pad(arr, (0, max_wav_len - len(arr)), constant_values=0) for arr in wavs])
        stacked = rearrange(stacked, "b n -> b 1 n")

        # breakpoint()
        return stacked
