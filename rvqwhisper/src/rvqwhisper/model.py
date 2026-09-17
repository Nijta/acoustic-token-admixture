from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from vector_quantize_pytorch import ResidualVQ


@dataclass
class Seq2SeqVQModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_features_quantized: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_hidden_states_before_quantization: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Seq2SeqVQLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_features_quantized: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_hidden_states_before_quantization: Optional[Tuple[torch.FloatTensor, ...]] = None

from torch.cuda.amp import autocast

class QuantizedWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig, vector_quantization_config):
        super().__init__(config)
        self.quantization_config = vector_quantization_config
        self.initialize_quantization(self.quantization_config)
        self.faster_whisper_model = None



    def initialize_quantization(self, quantization_config):
        self.quantization_loss = None
        self.quantization_config = quantization_config

        self.vector_quantization = ResidualVQ(
            dim=quantization_config["dim"],
            codebook_size=quantization_config["codebook_size"],
            num_quantizers=quantization_config["num_quantizers"],
            kmeans_init=True,  # set to True
            kmeans_iters=quantization_config[
                "init_iter"
            ],  # number of kmeans iterations to calculate the centroids for the codebook on init
        ).to("cuda")

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqVQModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:


            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask
            )
            encoder_outputs = self.faster_whisper_model.encode(input_features.detach().cpu().numpy())
            encoder_outputs = torch.asarray(encoder_outputs).to(self.device).half()
            # encoder_outputs = self.encoder(
            #     input_features,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                # last_hidden_state=encoder_outputs[0],
                last_hidden_state=encoder_outputs,
                # hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                hidden_states = None,
                # attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions= None,
            )

        encoder_hidden_states_before_quantization = encoder_outputs.last_hidden_state
        # breakpoint()
        # Apply vector quantization to the encoder outputs
        (
            encoder_outputs_quantized,
            quantization_indices,
            quantization_loss,
            all_encoder_outputs_quantized,
        ) = self.vector_quantization(
            encoder_hidden_states_before_quantization, return_all_codes=True
        )

        self.quantization_loss = quantization_loss.squeeze()

        # Replace the last hidden state with the quantized version
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs_quantized,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqVQModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_features_quantized=all_encoder_outputs_quantized,
            encoder_hidden_states_before_quantization=encoder_hidden_states_before_quantization,
        )


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class QuantizedWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig, vector_quantization_config):
        super().__init__(config)
        self.vector_quantization_config = vector_quantization_config
        self.model = QuantizedWhisperModel(
            config, vector_quantization_config=self.vector_quantization_config
        )
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def initialize_quantization(self, quantization_config):
        self.model.initialize_quantization(quantization_config)

    def pretrain_quantization(self, dataloader, quantization_config):
        self.model.pretrain_quantization(dataloader, quantization_config)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqVQLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_encoder_outputs_quantized = outputs.encoder_features_quantized
        encoder_hidden_states_before_quantization = (
            outputs.encoder_hidden_states_before_quantization
        )

        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            self.loss_asr = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1)
            )
            self.loss_qt = self.model.quantization_loss

            loss = self.loss_asr

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqVQLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_features_quantized=all_encoder_outputs_quantized,
            encoder_hidden_states_before_quantization=encoder_hidden_states_before_quantization,
        )
