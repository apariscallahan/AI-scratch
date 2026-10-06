"""Generate the example projects in neuroblocks/examples/ from a compact Python description.

Run:  python scripts/make_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroblocks.compiler import B  # noqa: E402
from neuroblocks.compiler.spec import _stack_json  # noqa: E402

OUT = ROOT / "neuroblocks" / "examples"


# --- tiny helpers for common reporter blocks -------------------------------------------
def num(v):
    return B("math_number", NUM=v)


def txt(v):
    return B("text", TEXT=v)


def var(name):
    return B("variables_get", VAR={"name": name})


def join(*parts):
    vals = {f"ADD{i}": (p if isinstance(p, B) else txt(str(p))) for i, p in enumerate(parts)}
    return B("text_join", _extra={"extraState": {"itemCount": len(parts)}}, **vals)


def mul(a, b):
    return B("math_arithmetic", OP="MULTIPLY", A=a if isinstance(a, B) else num(a), B=b if isinstance(b, B) else num(b))


def rnd(x, digits=1):
    return B("nb_round_to", NUM=x, DIGITS=digits)


def pct(score_block):
    return rnd(mul(score_block, 100), 1)


def lst(*items):
    vals = {f"ADD{i}": (p if isinstance(p, B) else num(p)) for i, p in enumerate(items)}
    return B("lists_create_with", _extra={"extraState": {"itemCount": len(items)}}, **vals)


def score(model, data, metric="accuracy"):
    return B("nb_score", METRIC=metric, MODEL=model, DATA=data)


def say(*parts):
    return B("nb_say", TEXT=join(*parts) if len(parts) > 1 else (parts[0] if isinstance(parts[0], B) else txt(parts[0])))


def stack(blocks, x=40, y=40):
    head = _stack_json(blocks)
    head["x"], head["y"] = x, y
    return head


def comment(text, w=380, h=120, below="output"):
    """A note, placed automatically below the output stack (or below / beside the training stack)."""
    return {"text": text, "width": w, "height": h, "_below": below}


def setv(name, value):
    return B("variables_set", VAR={"name": name}, VALUE=value)


def until(cond, body):
    return B("controls_whileUntil", MODE="UNTIL", BOOL=cond, DO=body)


def is_empty(value):
    return B("logic_compare", OP="EQ", A=value, B=txt(""))


def nums(*values):
    return lst(*values)


# --- training → output -------------------------------------------------------------------
def weights(model):
    """The 📦 block that ends a training stack (the output stack's wire starts here)."""
    return B("nb_weights", _id="weights", MODEL=model)


def output(*blocks):
    """An output stack, wired to the 📦 weights block."""
    return [B("nb_when_output", _id="output", PORT="weights")] + list(blocks)


def write_tokens(n, temp, top=5):
    """The language-model loop: score the next token, turn scores into probabilities, pick one, add it."""
    return B("controls_repeat_ext", TIMES=num(n), DO=[
        setv("scores", B("nb_o_next", TOKENS=var("tokens"))),
        setv("probs", B("nb_o_probs", SCORES=var("scores"), TEMP=temp)),
        setv("next token", B("nb_o_pick", HOW="random", PROBS=var("probs"))),
        B("nb_list_add", ITEM=var("next token"), VAR={"name": "tokens"}),
        B("nb_o_show_text", TEXT=B("nb_o_text", TOKENS=var("tokens"))),
        B("nb_o_show_top", N=top, PROBS=var("probs")),
    ])


def story_output(first_prompt, n, temp, question):
    return output(
        setv("prompt", txt(first_prompt)),
        until(is_empty(var("prompt")), [
            setv("tokens", B("nb_o_tokens", TEXT=var("prompt"))),
            write_tokens(n, temp),
            B("nb_o_ask", QUESTION=txt(question)),
            setv("prompt", B("nb_o_answer")),
        ]),
    )


def classify(input_block, n_top, what="It's"):
    """input → scores → probabilities → the most likely choice → its name."""
    return [
        setv("scores", B("nb_o_run", INPUT=input_block)),
        setv("probs", B("nb_o_probs", SCORES=var("scores"), TEMP=1)),
        B("nb_o_show_top", N=n_top, PROBS=var("probs")),
        say(f"{what} ", B("nb_o_name", CHOICE=B("nb_o_pick", HOW="best", PROBS=var("probs")))),
    ]


def drawing_output(question, times=5):
    return output(B("controls_repeat_ext", TIMES=num(times), DO=[
        B("nb_o_ask_draw", QUESTION=txt(question)),
        *classify(B("nb_o_answer"), 3, "I think that's a"),
    ]))


def play_output(world, result="score", play=False):
    """Sense → run the model → act, until the try is over (a trained policy driving)."""
    blocks = [
        B("nb_o_try", WORLD=world),
        until(B("nb_o_over"), [
            setv("senses", B("nb_o_senses")),
            setv("scores", B("nb_o_run", INPUT=var("senses"))),
            B("nb_o_do", ACTION=B("nb_o_action", SCORES=var("scores"))),
        ]),
        B("nb_o_try_show"),
        say(f"{'Distance' if result == 'distance' else 'Score'} of this try: ",
            rnd(B("nb_o_try_stat", WHAT=result), 1)),
    ]
    if play:
        blocks.append(B("nb_rl_play", WORLD=world))
    return output(*blocks)


EXAMPLES = []

# How big each example's stacks are when the editor draws them (workspace units): training (w, h) and
# output (w, h). The output stack goes to the right of the training stack, with room for the wire, and
# notes go underneath, so nothing overlaps. After changing an example a lot, re-measure in the editor:
# load it and read Blockly.getMainWorkspace().getTopBlocks().map(b => b.getBoundingRectangle()).
SIZES = {
    "01_spirals": (785, 1273, 633, 457), "02_iris_tree": (840, 1161, 622, 553), "03_curve": (402, 913, 962, 377),
    "04_lr_sweep": (1247, 1305, 0, 0), "05_digits_cnn": (528, 1073, 656, 481), "06_mnist": (493, 1121, 656, 481),
    "07_autoencoder": (555, 857, 694, 257), "07b_generate_digits": (557, 537, 579, 305),
    "08_tiny_gpt": (714, 729, 669, 761), "09_shakespeare_gpt": (652, 1121, 669, 761),
    "10_names": (460, 913, 669, 761), "11_sentiment": (607, 881, 656, 585), "12_finetune_gpt2": (620, 657, 669, 761),
    "13_compare_classic": (763, 1577, 0, 0), "14_clusters": (497, 697, 0, 0), "15_forecast": (542, 793, 0, 0),
    "20_car_ppo": (789, 1617, 522, 569), "21_car_evolution": (519, 1409, 522, 521), "22_cartpole": (312, 641, 600, 569),
    "23_maze": (308, 713, 600, 569), "24_lander": (297, 737, 600, 569), "25_flappy": (298, 689, 600, 569),
    "26_custom_reward": (925, 1153, 522, 521),
}
WIRE_GAP = 140  # room between the stacks for the wire


def _layout(id, comments, has_output):
    tw, th, ow, oh = SIZES.get(id, (700, 900, 600, 600))
    out_x = 40 + -(-(tw + WIRE_GAP) // 20) * 20
    free = {"output": [out_x, 40 + oh + 50], "training": [40, 40 + th + 50], "right": [40 + tw + 60, 40]}
    placed = []
    for c in comments:
        c = dict(c)
        where = c.pop("_below")
        if where == "output" and not has_output:
            where = "right"
        c["x"], c["y"] = free[where]
        free[where][1] += c["height"] + 30
        placed.append(c)
    return out_x, placed


def example(id, name, category, level, minutes, order, description, blocks, comments=(), extra_stacks=(),
            outputs=None):
    out_x, comments = _layout(id, comments, bool(outputs))
    tops = [stack(blocks)]
    if outputs:
        tops.append(stack(outputs, out_x, 40))
    ws = {"blocks": {"languageVersion": 0, "blocks": tops + list(extra_stacks)}}
    if comments:
        ws["workspaceComments"] = list(comments)
    EXAMPLES.append({"format": "neuroblocks-project", "version": 1, "name": name, "description": description,
                     "category": category, "level": level, "minutes": minutes, "order": order, "id": id,
                     "workspace": ws})


# =========================================================================================
# Start here
# =========================================================================================
example(
    "01_spirals", "Untangle two spirals", "Start here", "beginner", 1, 1,
    "A small neural network learns to separate two tangled spirals. Watch the decision map form while it trains.",
    [
        B("nb_when_run"),
        B("nb_data_toy", NAME="data", KIND="spirals",
          SETTINGS=[B("nb_ds_samples", N=600), B("nb_ds_noise", NOISE=0.15), B("nb_ds_test", PCT=20)]),
        B("nb_data_plot", DATA="data"),
        B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=32, ACT="relu"), B("nb_l_dense", UNITS=32, ACT="relu"),
                                            B("nb_l_dense", UNITS=32, ACT="relu"), B("nb_l_output")]),
        B("nb_train", MODEL="model", DATA="data", SETTINGS=[
            B("nb_t_epochs", N=150), B("nb_t_batch", N=32), B("nb_t_optimizer", OPT="adam", LR=0.01),
            B("nb_t_every", N=10, UNIT="epochs", DO=[B("nb_decision_map", MODEL="model", DATA="data")])]),
        B("nb_evaluate", MODEL="model", DATA="data"),
        say("Test accuracy: ", pct(score("model", "data")), "%"),
        B("nb_check", VALUE=score("model", "data"), OP=">=", TARGET=0.9),
        weights("model"),
    ],
    outputs=output(
        setv("point", nums(0.5, -0.5)),
        *classify(var("point"), 2, "The point (0.5, -0.5) belongs to the spiral:"),
    ),
    comments=[comment("Output: the trained network gets two numbers (a point), gives each spiral a score, the scores "
                      "become probabilities, and the most likely spiral wins. Change the point and run again!\n\n"
                      "Also try: more or fewer layers, tanh instead of ReLU, more noise, or the moons / XOR data.", w=380, h=170)],
)

example(
    "02_iris_tree", "Flowers & decision trees", "Start here", "beginner", 1, 2,
    "Classic machine learning: a decision tree learns yes/no questions that tell three kinds of iris flowers apart.",
    [
        B("nb_when_run"),
        B("nb_data_table", NAME="flowers", KIND="iris", SETTINGS=[B("nb_ds_test", PCT=25), B("nb_ds_standardize", ON=True)]),
        B("nb_data_plot", DATA="flowers"),
        B("nb_c_tree", NAME="tree", DEPTH=3),
        B("nb_train", MODEL="tree", DATA="flowers", SETTINGS=[]),
        B("nb_evaluate", MODEL="tree", DATA="flowers"),
        B("nb_c_show_tree", MODEL="tree"),
        B("nb_c_importance", MODEL="tree"),
        B("nb_c_knn", NAME="neighbours", K=5),
        B("nb_train", MODEL="neighbours", DATA="flowers", SETTINGS=[]),
        say("Decision tree: ", pct(score("tree", "flowers")), "%   ·   k-nearest neighbours: ",
            pct(score("neighbours", "flowers")), "%"),
        B("nb_decision_map", MODEL="tree", DATA="flowers"),
        B("nb_check", VALUE=score("tree", "flowers"), OP=">=", TARGET=0.85),
        weights("tree"),
    ],
    outputs=output(
        setv("flower", nums(5.1, 3.5, 1.4, 0.2)),
        *classify(var("flower"), 3, "This flower is probably a"),
    ),
    comments=[comment("The flower's four measurements (sepal length, sepal width, petal length, petal width) go "
                      "through the tree's questions. A tree is sure of itself, so one species gets almost 100%. "
                      "Try other measurements, e.g. 6.7, 3.0, 5.2, 2.3.", w=360, h=130)],
)

example(
    "03_curve", "Fit a curve (predict numbers)", "Start here", "beginner", 1, 3,
    "Regression: a network learns the shape of a noisy sine wave. The purple line is what it has learned.",
    [
        B("nb_when_run"),
        B("nb_data_toy", NAME="data", KIND="sine", SETTINGS=[B("nb_ds_samples", N=400), B("nb_ds_noise", NOISE=0.1)]),
        B("nb_data_plot", DATA="data"),
        B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=32, ACT="tanh"), B("nb_l_dense", UNITS=32, ACT="tanh"),
                                            B("nb_l_output")]),
        B("nb_train", MODEL="model", DATA="data", SETTINGS=[
            B("nb_t_epochs", N=120), B("nb_t_optimizer", OPT="adam", LR=0.01),
            B("nb_t_every", N=10, UNIT="epochs", DO=[B("nb_decision_map", MODEL="model", DATA="data")])]),
        B("nb_evaluate", MODEL="model", DATA="data"),
        weights("model"),
    ],
    outputs=output(
        B("controls_for", VAR={"name": "x"}, FROM=num(-3), TO=num(3), BY=num(0.25), DO=[
            B("nb_chart_point", X=var("x"), Y=B("nb_o_run", INPUT=var("x")), CHART=txt("What the model learned"),
              SERIES=txt("model's answer")),
        ]),
        say("For x = 1 the model answers ", rnd(B("nb_o_run", INPUT=num(1)), 3), " (the real curve: 0.997)"),
    ),
    comments=[comment("A model that predicts numbers answers with the number itself — no probabilities needed. "
                      "The loop asks it about x = -3 … 3 and draws its answers on the Charts tab.", w=360, h=110)],
)

example(
    "04_lr_sweep", "Experiment: which learning rate is best?", "Start here", "intermediate", 2, 4,
    "Loops + charts: train the same network with four learning rates and plot how well each one does.",
    [
        B("nb_when_run"),
        B("nb_data_toy", NAME="data", KIND="moons", SETTINGS=[B("nb_ds_samples", N=500), B("nb_ds_noise", NOISE=0.2)]),
        B("controls_forEach", VAR={"name": "lr"}, LIST=lst(1, 0.1, 0.01, 0.001), DO=[
            B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=16, ACT="relu"), B("nb_l_dense", UNITS=16, ACT="relu"),
                                                B("nb_l_output")]),
            B("nb_train", MODEL="model", DATA="data", SETTINGS=[
                B("nb_t_epochs", N=30), B("nb_t_optimizer", OPT="sgd", LR=var("lr")),
                B("nb_t_label", LABEL=join("lr ", var("lr")))]),
            B("nb_chart_point", X=B("math_single", OP="LOG10", NUM=var("lr")), Y=score("model", "data"),
              CHART=txt("Accuracy vs log10(learning rate)"), SERIES=txt("test accuracy")),
            say("learning rate ", var("lr"), " → test accuracy ", pct(score("model", "data")), "%"),
        ]),
    ],
    comments=[comment("Each pass of the loop builds a fresh network and trains it with a different learning rate. "
                      "Too big = chaos, too small = barely learns. Look at the loss chart: every run gets its own line.", w=330, h=130)],
)

# =========================================================================================
# Images
# =========================================================================================
example(
    "05_digits_cnn", "Read handwritten digits (CNN)", "Images", "beginner", 1, 10,
    "A convolutional network learns to read tiny 8×8 handwritten digits (built in, no download). Then draw your own!",
    [
        B("nb_when_run"),
        B("nb_data_images", NAME="digits", KIND="digits", SETTINGS=[B("nb_ds_augment", AUG="shift")]),
        B("nb_data_examples", N=24, DATA="digits"),
        B("nb_model", NAME="cnn", LAYERS=[
            B("nb_l_conv", FILTERS=32, SIZE=3, STRIDE=1, ACT="relu"), B("nb_l_conv", FILTERS=32, SIZE=3, STRIDE=1, ACT="relu"),
            B("nb_l_pool", KIND="max", SIZE=2), B("nb_l_flatten"), B("nb_l_dense", UNITS=64, ACT="relu"),
            B("nb_l_dropout", RATE=0.3), B("nb_l_output")]),
        B("nb_train", MODEL="cnn", DATA="digits", SETTINGS=[
            B("nb_t_epochs", N=25), B("nb_t_batch", N=32), B("nb_t_optimizer", OPT="adam", LR=0.002)]),
        B("nb_evaluate", MODEL="cnn", DATA="digits"),
        B("nb_show_predictions", MODEL="cnn", N=24, DATA="digits"),
        B("nb_check", VALUE=score("cnn", "digits"), OP=">=", TARGET=0.9),
        weights("cnn"),
    ],
    outputs=drawing_output("Draw a digit from 0 to 9 (big, in the middle)", 5),
    comments=[comment("Output: your drawing is shrunk to 8×8 pixels like the training pictures, the network gives "
                      "each digit 0–9 a score, and the scores become probabilities. Draw a sloppy 7 — is the model "
                      "still sure?", w=360, h=120)],
)

example(
    "06_mnist", "MNIST: the 'hello world' of deep learning", "Images", "intermediate", 3, 11,
    "70,000 real handwritten digits (downloaded once, ~11 MB). A classic CNN reaches ~99% accuracy. Then draw digits for it.",
    [
        B("nb_when_run"),
        B("nb_data_images", NAME="mnist", KIND="mnist", SETTINGS=[B("nb_ds_augment", AUG="shift")]),
        B("nb_data_examples", N=24, DATA="mnist"),
        B("nb_model", NAME="cnn", LAYERS=[
            B("nb_l_conv", FILTERS=32, SIZE=3, STRIDE=1, ACT="relu"), B("nb_l_pool", KIND="max", SIZE=2),
            B("nb_l_conv", FILTERS=64, SIZE=3, STRIDE=1, ACT="relu"), B("nb_l_pool", KIND="max", SIZE=2),
            B("nb_l_flatten"), B("nb_l_dense", UNITS=128, ACT="relu"), B("nb_l_dropout", RATE=0.3), B("nb_l_output")]),
        B("nb_train", MODEL="cnn", DATA="mnist", SETTINGS=[
            B("nb_t_epochs", N=3), B("nb_t_batch", N=64), B("nb_t_optimizer", OPT="adam", LR=0.001)]),
        B("nb_evaluate", MODEL="cnn", DATA="mnist"),
        B("nb_show_predictions", MODEL="cnn", N=32, DATA="mnist"),
        B("nb_save", MODEL="cnn", FILE=txt("mnist_cnn.pt")),
        weights("cnn"),
    ],
    outputs=drawing_output("Draw a digit from 0 to 9", 5),
    comments=[comment("Swap MNIST for Fashion-MNIST (clothes) or CIFAR-10 (colour photos) in the dropdown. CIFAR-10 "
                      "needs a bigger network and more epochs — a nice job for a cloud GPU (File ▸ Export cloud GPU bundle).", w=360, h=120)],
)

example(
    "07_autoencoder", "Squash & rebuild pictures (autoencoder)", "Images", "intermediate", 1, 12,
    "An autoencoder squeezes each picture through a tiny 8-number bottleneck and learns to rebuild it — no labels needed.",
    [
        B("nb_when_run"),
        B("nb_data_images", NAME="digits", KIND="digits", SETTINGS=[]),
        B("nb_model", NAME="autoencoder", LAYERS=[
            B("nb_l_flatten"), B("nb_l_dense", UNITS=48, ACT="relu"), B("nb_l_dense", UNITS=8, ACT="relu"),
            B("nb_l_dense", UNITS=48, ACT="relu"), B("nb_l_output")]),
        B("nb_train", MODEL="autoencoder", DATA="digits", SETTINGS=[
            B("nb_t_epochs", N=60), B("nb_t_optimizer", OPT="adam", LR=0.003), B("nb_t_goal", GOAL="reconstruct")]),
        B("nb_show_predictions", MODEL="autoencoder", N=16, DATA="digits"),
        B("nb_data_map", DATA="digits", METHOD="tsne"),
        weights("autoencoder"),
    ],
    outputs=output(B("controls_repeat_ext", TIMES=num(3), DO=[
        B("nb_o_ask_draw", QUESTION=txt("Draw a digit — it gets squeezed into 8 numbers and rebuilt")),
        B("nb_o_show_pic", PIC=B("nb_o_run", INPUT=B("nb_o_answer"))),
    ])),
)

example(
    "07b_generate_digits", "Draw new digits (generative AI)", "Images", "intermediate", 4, 13,
    "A picture generator studies 60,000 handwritten digits and learns to invent brand-new ones — you can even ask "
    "it for a particular digit. Watch the pictures sharpen while it trains.",
    [
        B("nb_when_run"),
        B("nb_data_images", NAME="mnist", KIND="mnist", SETTINGS=[]),
        B("nb_gen_create", NAME="artist", KIND="vae"),
        B("nb_train", MODEL="artist", DATA="mnist", SETTINGS=[
            B("nb_t_epochs", N=8), B("nb_t_batch", N=128)]),
        B("nb_gen_show", N=32, MODEL="artist", CLASS=txt("any")),
        B("nb_save", MODEL="artist", FILE=txt("digit_artist.pt")),
        weights("artist"),
    ],
    outputs=output(
        setv("noise", B("nb_o_noise")),
        B("controls_for", VAR={"name": "digit"}, FROM=num(0), TO=num(9), BY=num(1), DO=[
            B("nb_o_show_pic", PIC=B("nb_o_picture", NOISE=var("noise"), CLASS=var("digit"))),
        ]),
        say("Same noise, ten different digits: the noise decides the handwriting style. Run again for new noise!"),
    ),
    comments=[comment("Output: a generator turns a few random numbers (noise) into a picture. Here the SAME noise is "
                      "drawn as every digit 0–9, so all ten share one handwriting style.\n\n"
                      "Switch VAE to GAN for sharper (but less predictable) pictures — give a GAN more epochs.", w=370, h=150)],
)

# =========================================================================================
# Language
# =========================================================================================
example(
    "08_tiny_gpt", "Your first GPT (toy stories)", "Language models", "beginner", 2, 20,
    "Train a tiny GPT from scratch on simple made-up stories. Every few hundred steps it writes a sample — watch it "
    "go from gibberish to stories. Then prompt it yourself.",
    [
        B("nb_when_run"),
        B("nb_data_text", NAME="stories", KIND="toy_stories",
          SETTINGS=[B("nb_ds_tokenizer", TOK="characters"), B("nb_ds_context", N=64)]),
        B("nb_gpt", NAME="gpt", LAYERS=2, HEADS=4, EMBED=96, DROPOUT=0.0),
        B("nb_train", MODEL="gpt", DATA="stories", SETTINGS=[
            B("nb_t_steps", N=1500), B("nb_t_batch", N=32), B("nb_t_optimizer", OPT="adamw", LR=0.002),
            B("nb_t_every", N=300, UNIT="steps", DO=[
                B("nb_generate", LENGTH=150, MODEL="gpt", PROMPT=txt("Once upon a time"), TEMP=0.8)])]),
        weights("gpt"),
    ],
    outputs=story_output("One day, Max", 300, 0.7, "Start another story! Type its first words (or leave it empty "
                                                    "to stop)"),
    comments=[comment("How a GPT writes: turn the prompt into tokens (numbers), let the model score every possible "
                      "next token, turn the scores into probabilities, pick one at random, add it — and repeat. "
                      "Watch the bar chart: that's the model making up its mind, one character at a time.\n\n"
                      "Try: temperature 0.2 (safe, repetitive) or 1.5 (wild). 'pick the most likely' instead of "
                      "'a random' makes it always write the same thing.", w=420, h=190),
              comment("Training: a GPT reads characters and learns to guess the next one. 'layers' = depth, 'heads' = "
                      "how many things it pays attention to at once, 'embedding size' = width. Bigger = smarter but "
                      "slower.", w=380, h=120, below="training")],
)

example(
    "09_shakespeare_gpt", "Shakespeare GPT (scale it up on a GPU)", "Language models", "intermediate", 4, 21,
    "A character-level GPT trained on Shakespeare. Hyperparameter blocks let you scale it up from the command line "
    "on a cloud GPU without touching the blocks.",
    [
        B("nb_when_run"),
        B("nb_param", VAR={"name": "layers"}, VALUE=4),
        B("nb_param", VAR={"name": "embedding"}, VALUE=128),
        B("nb_param", VAR={"name": "context"}, VALUE=64),
        B("nb_param", VAR={"name": "batch"}, VALUE=16),
        B("nb_param", VAR={"name": "steps"}, VALUE=2000),
        B("nb_data_text", NAME="shakespeare", KIND="shakespeare",
          SETTINGS=[B("nb_ds_tokenizer", TOK="characters"), B("nb_ds_context", N=var("context"))]),
        B("nb_gpt", NAME="gpt", LAYERS=var("layers"), HEADS=4, EMBED=var("embedding"), DROPOUT=0.0),
        B("nb_show_model", MODEL="gpt"),
        B("nb_train", MODEL="gpt", DATA="shakespeare", SETTINGS=[
            B("nb_t_steps", N=var("steps")), B("nb_t_batch", N=var("batch")), B("nb_t_optimizer", OPT="adamw", LR=0.001),
            B("nb_t_schedule", SCHED="cosine", WARM=100),
            B("nb_t_every", N=500, UNIT="steps", DO=[
                B("nb_generate", LENGTH=200, MODEL="gpt", PROMPT=txt("ROMEO:"), TEMP=0.8)])]),
        B("nb_save", MODEL="gpt", FILE=txt("shakespeare_gpt.pt")),
        weights("gpt"),
    ],
    outputs=story_output("JULIET:", 500, 0.8, "Who speaks next? Type a name like HAMLET: (or leave it empty to stop)"),
    comments=[comment("These settings fit a laptop CPU. On a cloud GPU (File ▸ Export cloud GPU bundle), run:\n"
                      "bash run.sh --set layers=6 --set embedding=384 --set context=256 --set batch=64 "
                      "--set steps=5000\n"
                      "That is about the size of nanoGPT's Shakespeare model — it writes convincing fake Shakespeare. "
                      "The trained model comes back as models/shakespeare_gpt.pt.", w=400, h=190)],
)

example(
    "10_names", "Invent new names (build a GPT from layers)", "Language models", "intermediate", 2, 22,
    "The same idea as GPT, but built from individual layer blocks: the 'repeat' block sets how many transformer "
    "layers there are. It learns from 32,000 first names and invents new ones.",
    [
        B("nb_when_run"),
        B("nb_data_text", NAME="names", KIND="names", SETTINGS=[B("nb_ds_tokenizer", TOK="characters"),
                                                                 B("nb_ds_context", N=16)]),
        B("nb_model", NAME="namer", LAYERS=[
            B("nb_l_embedding", DIM=64), B("nb_l_posembed"),
            B("nb_l_repeat", TIMES=3, BODY=[B("nb_l_transformer", HEADS=4, DROP=0.1, MASK="auto")]),
            B("nb_l_layernorm"), B("nb_l_output")]),
        B("nb_train", MODEL="namer", DATA="names", SETTINGS=[
            B("nb_t_steps", N=2000), B("nb_t_batch", N=64), B("nb_t_optimizer", OPT="adamw", LR=0.003)]),
        weights("namer"),
    ],
    outputs=story_output("\n", 200, 0.9, "Type the first letters of a name, e.g. mar (or leave it empty to stop)"),
    comments=[comment("Every name in the training data ends with a new line (⏎), so the model learned to write "
                      "⏎ when a name is finished. Starting from ⏎ means: begin a brand-new name.", w=380, h=110)],
)

example(
    "11_sentiment", "Is this review positive? (text classifier)", "Language models", "beginner", 1, 23,
    "A small transformer reads short reviews and decides if they are positive or negative — including tricky ones "
    "like 'not boring at all'. Then type your own.",
    [
        B("nb_when_run"),
        B("nb_data_text_classes", NAME="reviews", KIND="toy_sentiment", SETTINGS=[B("nb_ds_tokenizer", TOK="words")]),
        B("nb_model", NAME="reader", LAYERS=[
            B("nb_l_embedding", DIM=32), B("nb_l_posembed"), B("nb_l_transformer", HEADS=2, DROP=0.1, MASK="auto"),
            B("nb_l_seqpool", MODE="mean"), B("nb_l_output")]),
        B("nb_train", MODEL="reader", DATA="reviews", SETTINGS=[
            B("nb_t_epochs", N=6), B("nb_t_batch", N=32), B("nb_t_optimizer", OPT="adam", LR=0.003)]),
        B("nb_evaluate", MODEL="reader", DATA="reviews"),
        B("nb_show_predictions", MODEL="reader", N=12, DATA="reviews"),
        weights("reader"),
    ],
    outputs=output(
        setv("review", txt("The movie was not boring at all")),
        until(is_empty(var("review")), [
            *classify(var("review"), 2, "I think this review is"),
            B("nb_o_ask", QUESTION=txt("Write your own movie review (or leave it empty to stop)")),
            setv("review", B("nb_o_answer")),
        ]),
    ),
    comments=[comment("The review is split into word tokens, the transformer reads them all at once and gives "
                      "'positive' and 'negative' a score each. Tricky ones: 'not bad', 'I expected to hate it'.", w=360, h=110)],
)

example(
    "12_finetune_gpt2", "Fine-tune a real GPT-2 (cloud GPU)", "Language models", "advanced", 15, 24,
    "Start from a pretrained GPT-2 from Hugging Face and fine-tune it on Shakespeare. Needs 'pip install transformers' "
    "and is best on a GPU — export it as a cloud bundle.",
    [
        B("nb_when_run"),
        B("nb_param", VAR={"name": "steps"}, VALUE=500),
        B("nb_data_text", NAME="shakespeare", KIND="shakespeare", SETTINGS=[B("nb_ds_context", N=128)]),
        B("nb_pretrained", MODELID=txt("distilgpt2"), NAME="gpt2"),
        B("nb_generate", LENGTH=80, MODEL="gpt2", PROMPT=txt("ROMEO:"), TEMP=0.8),
        B("nb_train", MODEL="gpt2", DATA="shakespeare", SETTINGS=[
            B("nb_t_steps", N=var("steps")), B("nb_t_batch", N=8), B("nb_t_optimizer", OPT="adamw", LR=0.00005)]),
        B("nb_save", MODEL="gpt2", FILE=txt("gpt2_shakespeare.pt")),
        weights("gpt2"),
    ],
    outputs=story_output("ROMEO:", 120, 0.8, "Type the start of a text (or leave it empty to stop)"),
    comments=[comment("GPT-2 knows 50,257 tokens — whole words and word pieces instead of single characters — so "
                      "the bar chart shows word pieces. The output loop is exactly the same as for a tiny GPT.", w=380, h=110)],
)

# =========================================================================================
# More machine learning
# =========================================================================================
example(
    "13_compare_classic", "Compare five classic ML models", "More ML", "beginner", 1, 30,
    "Trees, forests, k-NN, SVMs and logistic regression all try the same wine dataset. Which one wins?",
    [
        B("nb_when_run"),
        B("nb_data_table", NAME="wine", KIND="wine", SETTINGS=[B("nb_ds_test", PCT=25), B("nb_ds_standardize", ON=True)]),
        B("nb_c_tree", NAME="tree", DEPTH=4),
        B("nb_c_forest", NAME="forest", TREES=100, DEPTH=0),
        B("nb_c_knn", NAME="knn", K=5),
        B("nb_c_svm", NAME="svm", KERNEL="rbf"),
        B("nb_c_logistic", NAME="logistic"),
        B("nb_train", MODEL="tree", DATA="wine", SETTINGS=[]),
        B("nb_train", MODEL="forest", DATA="wine", SETTINGS=[]),
        B("nb_train", MODEL="knn", DATA="wine", SETTINGS=[]),
        B("nb_train", MODEL="svm", DATA="wine", SETTINGS=[]),
        B("nb_train", MODEL="logistic", DATA="wine", SETTINGS=[]),
        B("variables_set", VAR={"name": "scores"}, VALUE=lst(score("tree", "wine"), score("forest", "wine"),
                                                              score("knn", "wine"), score("svm", "wine"),
                                                              score("logistic", "wine"))),
        B("nb_chart_list", KIND="bar", LIST=var("scores"), TITLE=txt("Accuracy: 1 tree · 2 forest · 3 k-NN · 4 SVM · 5 logistic")),
        say("Best accuracy: ", pct(B("math_on_list", OP="MAX", LIST=var("scores"))), "%"),
        B("nb_c_importance", MODEL="forest"),
        B("nb_check", VALUE=score("forest", "wine"), OP=">=", TARGET=0.9),
    ],
)

example(
    "14_clusters", "Find groups without answers (k-means)", "More ML", "beginner", 1, 31,
    "Unsupervised learning: k-means finds groups in data all by itself. Then a t-SNE map shows how digit pictures "
    "cluster.",
    [
        B("nb_when_run"),
        B("nb_data_toy", NAME="dots", KIND="blobs", SETTINGS=[B("nb_ds_classes", N=4), B("nb_ds_samples", N=600),
                                                               B("nb_ds_noise", NOISE=0.3)]),
        B("nb_data_plot", DATA="dots"),
        B("nb_c_kmeans", NAME="groups", K=4),
        B("nb_train", MODEL="groups", DATA="dots", SETTINGS=[]),
        B("nb_decision_map", MODEL="groups", DATA="dots"),
        B("nb_data_images", NAME="digits", KIND="digits", SETTINGS=[]),
        B("nb_data_map", DATA="digits", METHOD="tsne"),
    ],
)

example(
    "15_forecast", "Predict what comes next (time series)", "More ML", "intermediate", 1, 32,
    "An LSTM looks at the last 48 values of a wavy signal and predicts the next one.",
    [
        B("nb_when_run"),
        B("nb_data_series", NAME="signal", KIND="seasonal", SETTINGS=[B("nb_ds_window", W=48, H=1), B("nb_ds_noise", NOISE=0.05)]),
        B("nb_model", NAME="forecaster", LAYERS=[B("nb_l_rnn", KIND="lstm", UNITS=64, KEEP="last"),
                                                 B("nb_l_dense", UNITS=32, ACT="relu"), B("nb_l_output")]),
        B("nb_train", MODEL="forecaster", DATA="signal", SETTINGS=[
            B("nb_t_epochs", N=8), B("nb_t_batch", N=32), B("nb_t_optimizer", OPT="adam", LR=0.003)]),
        B("nb_evaluate", MODEL="forecaster", DATA="signal"),
        B("nb_show_predictions", MODEL="forecaster", N=10, DATA="signal"),
    ],
)

# =========================================================================================
# Simulations (reinforcement learning)
# =========================================================================================
CAR_PARTS = [
    B("nb_w_terrain", KIND="rough", BUMP=0.5, HILLS=1.0, LEN=120),
    B("nb_w_carbody", LEN=2.0, WHEEL=0.45, W=1.0),
    B("nb_w_drive", WHEELS="rear", POWER=1.0),
    B("nb_w_lean", S=1.0),
    B("nb_w_sense", WHAT="speed"),
    B("nb_w_sense", WHAT="tilt"),
    B("nb_w_sense", WHAT="spin"),
    B("nb_w_sense_ground", RAYS=5, REACH=4.0),
    B("nb_w_sense", WHAT="wheels"),
    B("nb_w_reward", WHAT="distance", W=1.0),
    B("nb_w_finish", W=50),
    B("nb_w_flip", W=10, END=True),
    B("nb_w_stuck", SECS=3),
    B("nb_w_end_after", SECS=30),
    B("nb_w_same", ON=True),
]

example(
    "20_car_ppo", "Car on rough terrain (learns to drive)", "Simulations", "beginner", 2, 40,
    "Build a little car, give it abilities (drive, lean) and senses (speed, tilt, a ground radar), reward it for "
    "distance, and let reinforcement learning (PPO) figure out how to drive over the bumps. Then try it yourself!",
    [
        B("nb_when_run"),
        B("nb_world_car", NAME="world", PARTS=CAR_PARTS),
        B("nb_rl_show", WORLD="world"),
        B("nb_model", NAME="driver", LAYERS=[B("nb_l_dense", UNITS=64, ACT="tanh"), B("nb_l_dense", UNITS=64, ACT="tanh"),
                                             B("nb_l_output")]),
        B("nb_rl_train", MODEL="driver", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="ppo"), B("nb_rl_steps", N=150000), B("nb_rl_parallel", N=8),
            B("nb_rl_replay", N=5)]),
        say("Average score: ", rnd(B("nb_rl_score", MODEL="driver", WORLD="world", N=5), 1)),
        B("nb_save", MODEL="driver", FILE=txt("car_driver.pt")),
        weights("driver"),
    ],
    outputs=play_output("world", "distance", play=True),
    comments=[comment("Output: the trained driver senses the world (speed, tilt, ground radar …), runs those numbers "
                      "through its network to score each action, does the action, and repeats until the try is over. "
                      "Then it's your turn with the arrow keys!", w=380, h=130),
              comment("Things to try: make the terrain 'steps' or 'mixed', add 'can jump' or 'can rocket boost', "
                      "remove a sense and see if it still learns, change the rewards (reward speed? punish energy?), "
                      "or switch 'same world every try' off so it must handle any terrain.", w=380, h=130)],
)

example(
    "21_car_evolution", "Evolve a car (survival of the fittest)", "Simulations", "beginner", 2, 41,
    "A whole population of cars drives at once. The best ones become parents of the next generation, with small "
    "random mutations. Watch the ghosts get better generation after generation.",
    [
        B("nb_when_run"),
        B("nb_world_car", NAME="world", PARTS=[B("nb_w_terrain", KIND="mixed", BUMP=0.6, HILLS=1.0, LEN=150)]
          + CAR_PARTS[1:]),
        B("nb_model", NAME="driver", LAYERS=[B("nb_l_dense", UNITS=16, ACT="tanh"), B("nb_l_output")]),
        B("nb_rl_train", MODEL="driver", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="evolution"), B("nb_rl_generations", N=25), B("nb_rl_population", N=40),
            B("nb_rl_mutation", S=0.1), B("nb_rl_replay", N=2)]),
        weights("driver"),
    ],
    outputs=play_output("world", "distance"),
)

example(
    "22_cartpole", "Balance a pole", "Simulations", "beginner", 2, 42,
    "The classic control problem: push a cart left or right to keep the pole upright. PPO learns it in a minute or two. "
    "Can you do better with the arrow keys?",
    [
        B("nb_when_run"),
        B("nb_world_cartpole", NAME="world", PARTS=[B("nb_w_end_after", SECS=20)]),
        B("nb_model", NAME="balancer", LAYERS=[B("nb_l_dense", UNITS=64, ACT="tanh"), B("nb_l_dense", UNITS=64, ACT="tanh"),
                                               B("nb_l_output")]),
        B("nb_rl_train", MODEL="balancer", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="ppo"), B("nb_rl_steps", N=50000)]),
        weights("balancer"),
    ],
    outputs=play_output("world", "score", play=True),
)

example(
    "23_maze", "Escape the maze (Q-table)", "Simulations", "beginner", 1, 43,
    "Q-learning keeps a table of how good each move is in each square. Arrows in the replay show what it has learned.",
    [
        B("nb_when_run"),
        B("nb_world_maze", NAME="maze", PARTS=[B("nb_w_maze_size", W=8, H=8), B("nb_w_maze_walls", PCT=25),
                                               B("nb_w_maze_traps", N=3), B("nb_w_same", ON=True)]),
        B("nb_rl_show", WORLD="maze"),
        B("nb_model", NAME="explorer", LAYERS=[]),
        B("nb_rl_train", MODEL="explorer", WORLD="maze", SETTINGS=[
            B("nb_rl_algo", ALGO="qtable"), B("nb_rl_steps", N=30000)]),
        weights("explorer"),
    ],
    outputs=play_output("maze", "score", play=True),
    comments=[comment("A Q-table has no neural network: for every square it stores a score for each move (up, down, "
                      "left, right). 'run the model on' looks up the row for the square it senses, and the action "
                      "is the move with the highest score.", w=380, h=130)],
)

example(
    "24_lander", "Land the rocket", "Simulations", "intermediate", 8, 44,
    "Fire the main engine and side thrusters to land gently on the pad. A harder problem — give it time (or a GPU).",
    [
        B("nb_when_run"),
        B("nb_world_lander", NAME="world", PARTS=[B("nb_w_wind", W=0), B("nb_w_fuel", F=100)]),
        B("nb_model", NAME="pilot", LAYERS=[B("nb_l_dense", UNITS=64, ACT="tanh"), B("nb_l_dense", UNITS=64, ACT="tanh"),
                                            B("nb_l_output")]),
        B("nb_rl_train", MODEL="pilot", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="ppo"), B("nb_rl_steps", N=300000), B("nb_rl_parallel", N=8)]),
        weights("pilot"),
    ],
    outputs=play_output("world", "score", play=True),
)

example(
    "25_flappy", "Flappy bird", "Simulations", "beginner", 3, 45,
    "Flap through the gaps! Neuro-evolution is great at this one.",
    [
        B("nb_when_run"),
        B("nb_world_flappy", NAME="world", PARTS=[B("nb_w_gap", G=3.5), B("nb_w_pipe_speed", S=3)]),
        B("nb_model", NAME="bird", LAYERS=[B("nb_l_dense", UNITS=16, ACT="tanh"), B("nb_l_output")]),
        B("nb_rl_train", MODEL="bird", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="evolution"), B("nb_rl_generations", N=25), B("nb_rl_population", N=50)]),
        weights("bird"),
    ],
    outputs=play_output("world", "score", play=True),
)

example(
    "26_custom_reward", "Design your own reward (car)", "Simulations", "advanced", 5, 46,
    "Rewards decide what the car learns. Here it is rewarded for speed but punished for tilting — written as a formula "
    "with the 'world's …' blocks. Try your own ideas (e.g. reward jumping high!).",
    [
        B("nb_when_run"),
        B("nb_world_car", NAME="world", PARTS=[
            B("nb_w_terrain", KIND="hills", BUMP=0.3, HILLS=1.5, LEN=150),
            B("nb_w_drive", WHEELS="all", POWER=1.2), B("nb_w_jump", S=1.0),
            B("nb_w_sense", WHAT="speed"), B("nb_w_sense", WHAT="tilt"), B("nb_w_sense_ground", RAYS=7, REACH=5.0),
            B("nb_w_sense", WHAT="wheels"),
            B("nb_w_custom_reward", EXPR=B("math_arithmetic", OP="MINUS",
                                           A=mul(B("nb_w_state", WHAT="speed"), 0.1),
                                           B=mul(B("math_single", OP="ABS", NUM=B("nb_w_state", WHAT="tilt")), 0.002))),
            B("nb_w_flip", W=10, END=True),
            B("nb_w_end_when", COND=B("logic_compare", OP="GT", A=B("nb_w_state", WHAT="distance"), B=num(150))),
            B("nb_w_end_after", SECS=30)]),
        B("nb_model", NAME="driver", LAYERS=[B("nb_l_dense", UNITS=64, ACT="tanh"), B("nb_l_dense", UNITS=64, ACT="tanh"),
                                             B("nb_l_output")]),
        B("nb_rl_train", MODEL="driver", WORLD="world", SETTINGS=[
            B("nb_rl_algo", ALGO="ppo"), B("nb_rl_steps", N=150000)]),
        weights("driver"),
    ],
    outputs=play_output("world", "distance"),
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.nblk"):
        old.unlink()
    from neuroblocks.compiler import compile_workspace
    from neuroblocks.compiler.validate import check_connections
    bad = 0
    for ex in EXAMPLES:
        res = compile_workspace(ex["workspace"], markers=False, project=ex["name"])
        errs = [d.message for d in res.diagnostics if d.level == "error"]
        errs += [f"the editor would refuse to load it: {p}" for p in check_connections(ex["workspace"])]
        warns = [d.message for d in res.diagnostics if d.level == "warning"]
        status = "OK " if not errs else "ERR"
        if errs:
            bad += 1
        print(f"{status} {ex['id']:22s} {ex['name']}" + (f"  ({len(warns)} warnings)" if warns else ""))
        for m in errs:
            print(f"      error: {m}")
        for m in warns:
            print(f"      warning: {m}")
        (OUT / f"{ex['id']}.nblk").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(EXAMPLES)} examples written to {OUT}" + (f" — {bad} with errors" if bad else ""))
    return bad


if __name__ == "__main__":
    sys.exit(main())
