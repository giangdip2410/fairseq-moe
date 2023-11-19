# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
# from fairseq.models.Brainformer import (
#     DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, BrainformerDecoder
# )
from .brainformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, BrainformerDecoder
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class BrainformerLanguageModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Brainformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )

    # config for "MoE Layers"
    moe_type: Optional[str] = field(
        default='base_layer', metadata={"help": "what MoE layer to use, can be dense_base_layer, switch_layer, hash_layer, or base_layer"}
    )
    moe_layers: Optional[int] = field(
        default=0, metadata={"help": "number of MoE layers in total"}
    )
    moe_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each MoE layer"}
    )
    train_token_shuffle: Optional[int] = field(
        default=1, metadata={"help": "shuffle tokens between workers before computing assignment"}
    )
    assignment_algorithm: Optional[str] = field(
        default='BA', metadata={"help": "assignment algorithm to adopt"}
    )
    # config for two-stage training and distilled routing model
    two_stage_updates: Optional[int] = field(
        default=10000000,
        metadata={"help": "enter Stage-2 after how many updates"},
    )
    distill_assignment: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to distill the greedy assignment"},
    )
    distilled_model: Optional[str] = field(
        default='wordemb',
        metadata={"help": "what model to use for routing"},
    )
    #Optinal SMOE
    moe_expert_count: Optional[int] = field(
        default=16,
        metadata={"help": "MOE expert count"},
    )
    moe_freq: Optional[int] = field(
        default=1,
        metadata={"help": "moe-freq"},
    )
    moe_gating_use_fp32: Optional[int] = field(
        default=2,
        metadata={"help": "moe-gating-use-fp32"},
    )
    moe_second_expert_policy: Optional[str] = field(
        default="random",
        metadata={"help": "moe-second-expert-policy"},
    )
    moe_normalize_gate_prob_before_dropping: Optional[bool] = field(
        default=True,
        metadata={"help": "moe-normalize-gate-prob-before-dropping"},
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=-1.0,
        metadata={"help": "moe-eval-capacity-token-fraction"},
    )
    moe_gate_loss_wt: Optional[float] = field(
        default=0.01,
        metadata={"help": "moe-gate-loss-wt"},
    )
    moe_gate_loss_combine_method: Optional[str] = field(
        default='sum',
        metadata={"help": "moe-gate-loss-combine-method"},
    )
    moe_top1_expert: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use top1 gate instead of top2"
        }
    )


    # config for decomposed Brainformers (if distilled_model == 'trmxl')
    hf_plm_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the hugging face pretrained model"},
    )
    vocab_size: Optional[int] = field(
        default=50261,
        metadata={"help": "the vocabulary size"},
    )
    dict_pad_idx: Optional[int] = field(
        default=1,
        metadata={"help": "the pad index in the dictionary"},
    )
    dict_bos_idx: Optional[int] = field(
        default=0,
        metadata={"help": "the bos index in the dictionary"},
    )
    dict_eos_idx: Optional[int] = field(
        default=2,
        metadata={"help": "the eos index in the dictionary"},
    )
    # config for balance loss
    balance_loss: Optional[str] = field(
        default=None,
        metadata={"help": "which balance auxiliary loss to use"},
    )
    # config for Hash Layer
    hash_dict_path: Optional[str] = field(
        default='/home/v-damaidai/data/unilm/ddm/fairseq_pt/data-bin/hash_dict/hash_dict_8.json',
        metadata={"help": "the path to the hash dict"},
    )
    # config for Switch Layer
    capacity_factor: Optional[float] = field(
        default=100.0,
        metadata={"help": "capacity factor for Switch Layer"},
    )
    # config for DenseBaseLayer
    widex: Optional[int] = field(
        default=1, metadata={"help": "expand FFN width"}
    )
    deepx: Optional[int] = field(
        default=1, metadata={"help": "expand FFN stacked num"}
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")


@register_model("brainformer_lm_stable", dataclass=BrainformerLanguageModelConfig)
class BrainformerLanguageModel(FairseqLanguageModel):
    @classmethod
    def hub_models(cls):
        def moses_fastbpe(path):
            return {"path": path, "tokenizer": "moses", "bpe": "fastbpe"}

        def spm(path):
            return {"path": path, "tokenizer": "space", "bpe": "sentencepiece"}

        return {
            "brainformer_lm_stable.gbw.adaptive_huge": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2",
            "brainformer_lm_stable.wiki103.adaptive": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2",
            "brainformer_lm_stable.wmt19.en": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2"
            ),
            "brainformer_lm_stable.wmt19.de": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2"
            ),
            "brainformer_lm_stable.wmt19.ru": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2"
            ),
            "brainformer_lm_stable.wmt20.en": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.en.tar.gz"
            ),
            "brainformer_lm_stable.wmt20.ta": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.ta.tar.gz"
            ),
            "brainformer_lm_stable.wmt20.iu.news": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.news.tar.gz"
            ),
            "brainformer_lm_stable.wmt20.iu.nh": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.nh.tar.gz"
            ),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = BrainformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--activation-fn',
    #                         choices=utils.get_available_activation_fns(),
    #                         help='activation function to use')
    #     parser.add_argument('--dropout', type=float, metavar='D',
    #                         help='dropout probability')
    #     parser.add_argument('--attention-dropout', type=float, metavar='D',
    #                         help='dropout probability for attention weights')
    #     parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
    #                         help='dropout probability after activation in FFN.')
    #     parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained encoder embedding')
    #     parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension')
    #     parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension for FFN')
    #     parser.add_argument('--encoder-layers', type=int, metavar='N',
    #                         help='num encoder layers')
    #     parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
    #                         help='num encoder attention heads')
    #     parser.add_argument('--encoder-normalize-before', action='store_true',
    #                         help='apply layernorm before each encoder block')
    #     parser.add_argument('--encoder-learned-pos', action='store_true',
    #                         help='use learned positional embeddings in the encoder')
    #     parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained decoder embedding')
    #     parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension')
    #     parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension for FFN')
    #     parser.add_argument('--decoder-layers', type=int, metavar='N',
    #                         help='num decoder layers')
    #     parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
    #                         help='num decoder attention heads')
    #     parser.add_argument('--decoder-learned-pos', action='store_true',
    #                         help='use learned positional embeddings in the decoder')
    #     parser.add_argument('--decoder-normalize-before', action='store_true',
    #                         help='apply layernorm before each decoder block')
    #     parser.add_argument('--decoder-output-dim', type=int, metavar='N',
    #                         help='decoder output dimension (extra linear layer '
    #                              'if different from decoder embed dim')
    #     parser.add_argument('--share-decoder-input-output-embed', action='store_true',
    #                         help='share decoder input and output embeddings')
    #     parser.add_argument('--share-all-embeddings', action='store_true',
    #                         help='share encoder, decoder and output embeddings'
    #                              ' (requires shared dictionary and embed dim)')
    #     parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
    #                         help='if set, disables positional embeddings (outside self attention)')
    #     parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
    #                         help='comma separated list of adaptive softmax cutoff points. '
    #                              'Must be used with adaptive_loss criterion'),
    #     parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
    #                         help='sets adaptive softmax dropout for the tail projections')
    #     parser.add_argument('--layernorm-embedding', action='store_true',
    #                         help='add layernorm to embedding')
    #     parser.add_argument('--no-scale-embedding', action='store_true',
    #                         help='if True, dont scale embeddings')
    #     parser.add_argument('--checkpoint-activations', action='store_true',
    #                         help='checkpoint activations at each layer, which saves GPU '
    #                              'memory usage at the cost of some additional compute')
    #     parser.add_argument('--offload-activations', action='store_true',
    #                         help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
    #     # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    #     parser.add_argument('--no-cross-attention', default=False, action='store_true',
    #                         help='do not perform cross-attention')
    #     parser.add_argument('--cross-self-attention', default=False, action='store_true',
    #                         help='perform cross+self-attention')
    #     # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    #     parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
    #                         help='LayerDrop probability for encoder')
    #     parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
    #                         help='LayerDrop probability for decoder')
    #     parser.add_argument('--encoder-layers-to-keep', default=None,
    #                         help='which layers to *keep* when pruning as a comma-separated list')
    #     parser.add_argument('--decoder-layers-to-keep', default=None,
    #                         help='which layers to *keep* when pruning as a comma-separated list')
    #     # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    #     parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
    #                         help='iterative PQ quantization noise at training time')
    #     parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
    #                         help='block size of quantization noise at training time')
    #     parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
    #                         help='scalar quantization noise and scalar quantization at training time')
    #     # args for Fully Sharded Data Parallel (FSDP) training
    #     parser.add_argument(
    #         '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
    #         help=(
    #             'minimum number of params for a layer to be wrapped with FSDP() when '
    #             'training with --ddp-backend=fully_sharded. Smaller values will '
    #             'improve memory efficiency, but may make torch.distributed '
    #             'communication less efficient due to smaller input sizes. This option '
    #             'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
    #             '--offload-activations are passed.'
    #         )
    #     )
    #     # args for mixture-of-expert layers
    #     parser.add_argument('--moe-freq', type=int, metavar='D', default=0,
    #                         help='Frequency at which we insert MoE Transformer layers')
    #     parser.add_argument('--encoder-moe-freq', type=int, metavar='D', default=0,
    #                         help='Frequency at which we insert MoE Transformer encoder layers')
    #     parser.add_argument('--decoder-moe-freq', type=int, metavar='D', default=0,
    #                         help='Frequency at which we insert MoE Transformer decoder layers')
    #     parser.add_argument('--moe-expert-count', type=int, metavar='D', default=0,
    #                         help='Number of experts in each MoE Layer')
    #     parser.add_argument('--moe-gating-use-fp32', default=False, action='store_true',
    #                         help="Use FP32 computations in MoE top2 gating function")
    #     parser.add_argument('--moe-second-expert-policy', type=str, default='sampling',
    #                         help="policy for second expert, options: all/sampling/random")
    #     parser.add_argument('--moe-normalize-gate-prob-before-dropping', default=False, action='store_true',
    #                         help="whether to normalize gate probs before or after dropping experts for capacity and randomization")
    #     parser.add_argument('--moe-expert-ffn-dim', type=int, default=0,
    #                         help="MoE Expert FFN dimension")
    #     parser.add_argument('--moe-top1-expert', default=False, action='store_true',
    #                         help="Use top1 gate instead of top2")
    #     parser.add_argument('--encoder-moe-top1-expert', default=False, action='store_true',
    #                         help="Use top1 gate instead of top2 in encoder")
    #     parser.add_argument('--decoder-moe-top1-expert', default=False, action='store_true',
    #                         help="Use top1 gate instead of top2 in decoder")
    #     parser.add_argument('--moe-eval-capacity-token-fraction', type=float, default=0.25,
    #                         help="Fraction of tokens as capacity during validation" + \
    #                              "if set to negative, use same as training. range: (0.0, 1.0].")
    #     parser.add_argument('--capacity-factor', type=float, default=1.0,
    #                         help="Fraction of tokens as capacity during training")
    #     parser.add_argument('--moe-normalize-expert-grad', type=str, default='world_size',
    #                         help="Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'")
    #     parser.add_argument('--use-moe-pad-mask', default=False, action='store_true',
    #                         help="Don't route padding tokens to any expert")
    #     parser.add_argument('--moe-batch-prioritized-routing', default=False, action='store_true',
    #                         help="if true orders token by the gate prob before capacity dropping.")
    #     parser.add_argument('--dummy-a2a', default=False, action='store_true',
    #                         help="if true do not do all2all communication")
    #     # args for pseudo-MoE layers
    #     parser.add_argument('--alternate-ffn-embed-dim', type=int, default=0,
    #                         help="FFN embed dim of alternate pseudo-MoE blocks")
    #     parser.add_argument('--scale-for-route', type=float, default=1.0,
    #                         help="scale to tune the value of embedding before routing")
    #     parser.add_argument('--use-tutel', default=False, action='store_true',
    #                         help="if true use tutel impl")
    #     parser.add_argument('--use-tutel-all2all', default=False, action='store_true',
    #                         help="if true use all2all of tutel impl")
    #     parser.add_argument('--record-a2a-perf-stats', default=False, action='store_true',
    #                         help="if true record-a2a-perf-stats")
    #     parser.add_argument('--record-token-expert', default=False, action='store_true',
    #                         help="if true record token expert relations")
    #     # fmt: on


    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.moe_layers = getattr(args, "moe_layers", 0)
    args.moe_sublayers = getattr(args, "moe_sublayers", 1)
    args.train_token_shuffle = getattr(args, "train_token_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_tiny")
def brainformer_lm_small(args):
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_small")
def brainformer_lm_small(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_medium")
def brainformer_lm_medium(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_large")
def brainformer_lm_large(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 784)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)



@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_big")
def brainformer_lm_stable_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_wiki103")
@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_baevski_wiki103")
def brainformer_lm_stable_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    brainformer_lm_stable_big(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gbw")
@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_baevski_gbw")
def brainformer_lm_stable_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    brainformer_lm_stable_big(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt")
def brainformer_lm_stable_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt2_small")
def brainformer_lm_stable_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt2_tiny")
def brainformer_lm_stable_gpt2_tiny(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt2_medium")
def brainformer_lm_stable_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt2_big")
def brainformer_lm_stable_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 25)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_BaseGPT_x1_large")
def brainformer_lm_stable_BaseGPT_x1_large(args):
    # GPT2
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6144)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    # Base Layers
    args.moe_layers = getattr(args, "moe_layers", 1)
    args.moe_sublayers = getattr(args, "moe_sublayers", 3)
    args.train_token_shuffle = getattr(args, "train_token_shuffle", True)
    # general LM
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_BaseGPT_x1_medium")
def brainformer_lm_stable_BaseGPT_x1_medium(args):
    # GPT2
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    # Base Layers
    args.moe_layers = getattr(args, "moe_layers", 1)
    args.moe_sublayers = getattr(args, "moe_sublayers", 6)
    args.train_token_shuffle = getattr(args, "train_token_shuffle", True)
    # general LM
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_BaseGPT_x1_small")
def brainformer_lm_stable_BaseGPT_x1_small(args):
    # GPT2
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    # Base Layers
    args.moe_layers = getattr(args, "moe_layers", 1)
    args.moe_sublayers = getattr(args, "moe_sublayers", 3)
    args.train_token_shuffle = getattr(args, "train_token_shuffle", True)
    # general LM
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    # GPT-3 used learned positional embeddings, rather than sinusoidal
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.share_decoder_input_output_embed = True
    base_lm_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_small")
def brainformer_lm_stable_gpt3_small(args):
    # 125M params
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_medium")
def brainformer_lm_stable_gpt3_medium(args):
    # 350M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_large")
def brainformer_lm_stable_gpt3_large(args):
    # 760M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_xl")
def brainformer_lm_stable_gpt3_xl(args):
    # 1.3B params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_2_7")
def brainformer_lm_stable_gpt3_2_7(args):
    # 2.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_6_7")
def brainformer_lm_stable_gpt3_6_7(args):
    # 6.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_13")
def brainformer_lm_stable_gpt3_13(args):
    # 13B params
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 40)
    base_gpt3_architecture(args)


@register_model_architecture("brainformer_lm_stable", "brainformer_lm_stable_gpt3_175")
def brainformer_lm_stable_gpt3_175(args):
    # 175B params
    args.decoder_layers = getattr(args, "decoder_layers", 96)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 12288)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 96)
    base_gpt3_architecture(args)