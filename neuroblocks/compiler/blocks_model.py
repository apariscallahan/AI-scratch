"""Neural network blocks: models and layers."""
from __future__ import annotations

from .core import ORDER_FUNCTION_CALL, BlockCtx
from .helpers import assign, fmt_call, fmt_list, int_code
from .spec import LAYER, Drop, Name, Num, Ref, Stack, Str, block

ACT_OPTIONS = [("ReLU", "relu"), ("GELU", "gelu"), ("tanh", "tanh"), ("sigmoid", "sigmoid"),
               ("leaky ReLU", "leaky_relu"), ("SiLU (swish)", "silu"), ("none", "none")]


def _layer(b: BlockCtx, cls: str, *args: str, **kwargs: str) -> str:
    parts = [a for a in args if a is not None]
    parts += [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    parts += b.bid_kw()
    return f"nb.layers.{cls}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@block("nb_model", "neural", "create neural network %NAME %LAYERS",
       NAME=Name("model", "model"), LAYERS=Stack(LAYER), defines=("model", "NAME"), section="Build a model",
       prefill={"LAYERS": [("nb_l_dense", {"UNITS": 64, "ACT": "relu"}), ("nb_l_dense", {"UNITS": 64, "ACT": "relu"}),
                           "nb_l_output"]},
       tooltip="A neural network: stack layer blocks inside, top (input side) to bottom (output side).",
       help="Data flows from the top layer to the bottom one. You never have to say how big the input is — "
            "it's worked out from the data. End with an 'output layer'; it sizes itself to the number of "
            "classes (or numbers, or tokens) to predict.")
def _model(b):
    layers = b.layers("LAYERS")
    expr = fmt_call("nb.Model", [repr(b.name()), fmt_list(layers)] + b.bid_kw())
    return assign(b.name_ident(), expr)


@block("nb_gpt", "neural", "create GPT language model %NAME\nlayers %LAYERS heads %HEADS embedding size %EMBED "
                           "dropout %DROPOUT",
       NAME=Name("model", "model"), LAYERS=Num(4, integer=True), HEADS=Num(4, integer=True),
       EMBED=Num(128, integer=True), DROPOUT=Num(0.1), defines=("model", "NAME"),
       tooltip="A ready-made transformer like the ones behind ChatGPT — just smaller. Train it on a text dataset.",
       help="layers = how many transformer blocks are stacked (depth). heads = how many things each layer can "
            "pay attention to at once. embedding size = how many numbers describe each token (width; must "
            "divide by heads). More of each = smarter but slower. A cloud GPU can handle 8–12 layers with "
            "embedding 384–768.")
def _gpt(b):
    return assign(b.name_ident(), fmt_call("nb.GPT", [repr(b.name()), f"layers={int_code(b, 'LAYERS')}",
                                                      f"heads={int_code(b, 'HEADS')}",
                                                      f"embed={int_code(b, 'EMBED')}",
                                                      f"dropout={b.val('DROPOUT')}"] + b.bid_kw()))


@block("nb_pretrained", "neural", "load pretrained language model %MODELID as %NAME",
       MODELID=Str("distilgpt2"), NAME=Name("model", "model"), defines=("model", "NAME"),
       tooltip="Download a pretrained model from Hugging Face (e.g. distilgpt2, gpt2) to fine-tune on your text. "
               "Needs: pip install transformers. Best on a GPU.")
def _pretrained(b):
    return assign(b.name_ident(), fmt_call("nb.pretrained_lm", [b.val("MODELID"), f"name={b.name()!r}"] + b.bid_kw()))


@block("nb_gen_create", "neural", "create picture generator %NAME using %KIND", NAME=Name("model", "artist"),
       KIND=Drop([("VAE — steady, a bit blurry", "vae"), ("GAN — sharper, trickier", "gan")], "vae"),
       defines=("model", "NAME"), section="Generate pictures",
       tooltip="A model that learns to draw NEW pictures like the ones in an image dataset. Train it, then use "
               "'show new pictures from …'.",
       help="A VAE squeezes pictures into a few numbers and learns to turn random numbers back into pictures. "
            "A GAN has two players: an artist that draws and a critic that tries to spot fakes; they both get "
            "better. Both learn the classes too, so you can ask for 'a 7'.")
def _gen_create(b):
    return assign(b.name_ident(), fmt_call("nb.generator", [repr(b.name()), f"kind={b.q('KIND')}"] + b.bid_kw()))


@block("nb_show_model", "neural", "show %MODEL", MODEL=Ref("model", "model"), section="Look inside",
       tooltip="Show the model's layers, their sizes and parameter counts in the Model tab.")
def _show_model(b):
    return [f"nb.show_model({b.ref('MODEL')})"]


@block("nb_reset_model", "neural", "reset %MODEL", MODEL=Ref("model", "model"),
       tooltip="Forget everything the model learned (it starts again from random weights).")
def _reset(b):
    return [f"nb.reset({b.ref('MODEL')})"]


@block("nb_model_params", "neural", "number of parameters in %MODEL", shape="value", output="Number",
       MODEL=Ref("model", "model"), tooltip="How many numbers (weights) the model learns.")
def _params(b):
    return f"nb.param_count({b.ref('MODEL')})", ORDER_FUNCTION_CALL


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@block("nb_l_dense", "layers", "dense layer %UNITS units %ACT", shape="layer",
       UNITS=Num(64, integer=True), ACT=Drop(ACT_OPTIONS, "relu"), section="Basic",
       tooltip="Every input connects to every one of its neurons. The workhorse layer.",
       help="A dense (fully connected) layer has 'units' neurons. Each neuron adds up all its inputs times "
            "learned weights, then applies the activation (ReLU keeps positives, zeroes negatives) so the "
            "network can learn curvy shapes.")
def _dense(b):
    return _layer(b, "Dense", int_code(b, "UNITS"), activation=b.q("ACT"))


@block("nb_l_output", "layers", "output layer (sized automatically)", shape="layer",
       tooltip="The last layer: one unit per class / number / token. Its size is set from the data.")
def _output(b):
    return _layer(b, "Output")


@block("nb_l_activation", "layers", "activation %ACT", shape="layer", ACT=Drop(ACT_OPTIONS[:-1], "relu"),
       tooltip="Apply a non-linear function to every number.")
def _act(b):
    return _layer(b, "Activation", b.q("ACT"))


@block("nb_l_dropout", "layers", "dropout %RATE", shape="layer", RATE=Num(0.2),
       tooltip="While training, randomly switch off this fraction of signals so the network doesn't over-rely on "
               "any one of them (reduces overfitting).")
def _dropout(b):
    return _layer(b, "Dropout", b.val("RATE"))


@block("nb_l_batchnorm", "layers", "batch normalization", shape="layer",
       tooltip="Re-centre and re-scale signals using statistics of each batch — often trains faster.")
def _bn(b):
    return _layer(b, "BatchNorm")


@block("nb_l_layernorm", "layers", "layer normalization", shape="layer",
       tooltip="Re-centre and re-scale each example's signals (used in transformers).")
def _ln(b):
    return _layer(b, "LayerNorm")


@block("nb_l_flatten", "layers", "flatten", shape="layer",
       tooltip="Turn an image (channels × height × width) into one long list of numbers.")
def _flatten(b):
    return _layer(b, "Flatten")


@block("nb_l_conv", "layers", "conv 2D %FILTERS filters size %SIZE stride %STRIDE %ACT", shape="layer",
       FILTERS=Num(32, integer=True), SIZE=Num(3, integer=True), STRIDE=Num(1, integer=True),
       ACT=Drop(ACT_OPTIONS, "relu"), section="Images",
       tooltip="Slides small learned filters over the image to detect patterns like edges and shapes.",
       help="Each filter is a small window (e.g. 3×3) of learned weights that scans across the picture. Early "
            "conv layers find edges; deeper ones find eyes, wheels, letters… 'filters' = how many different "
            "patterns to look for. Stride 2 halves the image size.")
def _conv(b):
    return _layer(b, "Conv2D", int_code(b, "FILTERS"), size=int_code(b, "SIZE"), stride=int_code(b, "STRIDE"),
                  activation=b.q("ACT"))


@block("nb_l_pool", "layers", "%KIND pool %SIZE", shape="layer",
       KIND=Drop([("max", "max"), ("average", "average")], "max"), SIZE=Num(2, integer=True),
       tooltip="Shrink the image by keeping the biggest (or average) value of each little square.")
def _pool(b):
    return _layer(b, "Pool", b.q("KIND"), int_code(b, "SIZE"))


@block("nb_l_global_pool", "layers", "global %KIND pool", shape="layer",
       KIND=Drop([("average", "average"), ("max", "max")], "average"),
       tooltip="Squash each channel of an image (or each feature of a sequence) to a single number.")
def _gpool(b):
    return _layer(b, "GlobalPool", b.q("KIND"))


@block("nb_l_upsample", "layers", "upsample ×%SCALE", shape="layer", SCALE=Num(2, integer=True),
       tooltip="Make the image bigger (each pixel becomes a square) — used in decoders / generators.")
def _upsample(b):
    return _layer(b, "Upsample", int_code(b, "SCALE"))


@block("nb_l_reshape", "layers", "reshape to %SHAPE", shape="layer", SHAPE=Str("16,7,7"),
       tooltip="Rearrange numbers into a new shape, e.g. 16,7,7 = 16 channels of 7×7 pictures.")
def _reshape(b):
    return _layer(b, "Reshape", b.val("SHAPE"))


@block("nb_l_embedding", "layers", "token embedding %DIM numbers per token", shape="layer",
       DIM=Num(128, integer=True), section="Text & sequences",
       tooltip="Turn each token (character/word id) into a list of learned numbers. Start text models with this.")
def _emb(b):
    return _layer(b, "TokenEmbedding", int_code(b, "DIM"))


@block("nb_l_posembed", "layers", "positional embedding", shape="layer",
       tooltip="Tell the model where each token is in the sequence (transformers need this).")
def _posemb(b):
    return _layer(b, "PositionalEmbedding")


@block("nb_l_transformer", "layers", "transformer block %HEADS heads dropout %DROP %MASK", shape="layer",
       HEADS=Num(4, integer=True), DROP=Num(0.1),
       MASK=Drop([("auto", "auto"), ("can't see the future", "causal"), ("sees everything", "full")], "auto"),
       tooltip="Self-attention + a small network: each token looks at the other tokens to decide what matters.",
       help="The building block of GPT, BERT and friends. 'heads' are parallel attention patterns. For "
            "language models (auto) a token may only look at earlier tokens — it can't peek at the answer. "
            "Put several in a 'repeat' block to make the model deeper.")
def _transformer(b):
    return _layer(b, "TransformerBlock", heads=int_code(b, "HEADS"), dropout=b.val("DROP"),
                  mask=b.q("MASK") if b.field("MASK") != "auto" else None)


@block("nb_l_rnn", "layers", "%KIND layer %UNITS units %KEEP", shape="layer",
       KIND=Drop([("LSTM", "lstm"), ("GRU", "gru"), ("simple RNN", "rnn")], "lstm"), UNITS=Num(128, integer=True),
       KEEP=Drop([("keep all steps", "all"), ("keep last step only", "last")], "all"),
       tooltip="A recurrent layer reads a sequence step by step, carrying a memory along.")
def _rnn(b):
    return _layer(b, "Recurrent", b.q("KIND"), int_code(b, "UNITS"), keep=b.q("KEEP"))


@block("nb_l_seqpool", "layers", "combine steps by %MODE", shape="layer",
       MODE=Drop([("average", "mean"), ("max", "max"), ("last", "last"), ("first", "first")], "mean"),
       tooltip="Turn a sequence of vectors into one vector (e.g. before the output layer of a text classifier).")
def _seqpool(b):
    return _layer(b, "SeqPool", b.q("MODE"))


@block("nb_l_repeat", "layers", "repeat %TIMES times %BODY", shape="layer", TIMES=Num(4, integer=True),
       BODY=Stack(LAYER), section="Structure",
       prefill={"BODY": [("nb_l_transformer", {"HEADS": 4})]},
       tooltip="Stack copies of the layers inside (each copy learns its own weights). Great for deep networks.")
def _repeat(b):
    return f"nb.layers.Repeat({int_code(b, 'TIMES')}, {fmt_list(b.layers('BODY'))}{b.bid()})"


@block("nb_l_residual", "layers", "residual shortcut around %BODY", shape="layer", BODY=Stack(LAYER),
       prefill={"BODY": [("nb_l_dense", {"UNITS": 64, "ACT": "relu"})]},
       tooltip="Add the input back to the output of the layers inside (a 'skip connection', as in ResNet). "
               "Helps very deep networks train.")
def _residual(b):
    return f"nb.layers.Residual({fmt_list(b.layers('BODY'))}{b.bid()})"
