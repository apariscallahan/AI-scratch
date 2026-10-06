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


def comment(text, x, y, w=300, h=120):
    return {"text": text, "x": x, "y": y, "width": w, "height": h}


EXAMPLES = []


def example(id, name, category, level, minutes, order, description, blocks, comments=(), extra_stacks=()):
    ws = {"blocks": {"languageVersion": 0, "blocks": [stack(blocks)] + list(extra_stacks)}}
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
    ],
    comments=[comment("Try this: change the number of layers or units, the activation (ReLU → tanh), the noise, or the "
                      "dataset (moons, circles, XOR…). Then press Run again and compare the decision maps!", 720, 40, 320, 130)],
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
    ],
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
        say("A prediction for x = 1: ", B("nb_predict", MODEL="model", INPUT=lst(1))),
    ],
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
                      "Too big = chaos, too small = barely learns. Look at the loss chart: every run gets its own line.",
                      760, 40, 330, 130)],
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
        B("nb_play_draw", MODEL="cnn"),
    ],
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
        B("nb_play_draw", MODEL="cnn"),
    ],
    comments=[comment("Swap MNIST for Fashion-MNIST (clothes) or CIFAR-10 (colour photos) in the dropdown. CIFAR-10 "
                      "needs a bigger network and more epochs — a nice job for a cloud GPU (File ▸ Export cloud GPU bundle).",
                      760, 40, 330, 140)],
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
    ],
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
        B("nb_gen_show", N=16, MODEL="artist", CLASS=txt("7")),
        B("nb_save", MODEL="artist", FILE=txt("digit_artist.pt")),
    ],
    comments=[comment("Switch VAE to GAN for sharper (but less predictable) pictures — give a GAN more epochs. "
                      "Try Fashion-MNIST to invent clothes, or your own image folder!", 720, 40, 320, 110)],
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
        B("nb_generate", LENGTH=400, MODEL="gpt", PROMPT=txt("One day, Max"), TEMP=0.7),
        B("nb_play_chat", MODEL="gpt"),
    ],
    comments=[comment("A GPT reads characters and learns to guess the next one. 'layers' = depth, 'heads' = how many "
                      "things it pays attention to at once, 'embedding size' = width. Bigger = smarter but slower.",
                      760, 40, 330, 130)],
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
        B("nb_generate", LENGTH=600, MODEL="gpt", PROMPT=txt("JULIET:"), TEMP=0.8),
        B("nb_save", MODEL="gpt", FILE=txt("shakespeare_gpt.pt")),
        B("nb_play_chat", MODEL="gpt"),
    ],
    comments=[comment("These settings fit a laptop CPU. On a cloud GPU (File ▸ Export cloud GPU bundle), run:\n"
                      "bash run.sh --set layers=6 --set embedding=384 --set context=256 --set batch=64 "
                      "--set steps=5000\n"
                      "That is about the size of nanoGPT's Shakespeare model — it writes convincing fake Shakespeare. "
                      "The trained model comes back as models/shakespeare_gpt.pt.", 780, 40, 380, 190)],
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
        B("nb_generate", LENGTH=300, MODEL="namer", PROMPT=txt("\n"), TEMP=0.9),
        B("nb_generate", LENGTH=120, MODEL="namer", PROMPT=txt("\nmar"), TEMP=0.8),
    ],
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
        say("“The movie was not boring at all” → ",
            B("nb_predict", MODEL="reader", INPUT=txt("The movie was not boring at all"))),
        B("nb_play_form", MODEL="reader"),
    ],
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
        B("nb_generate", LENGTH=200, MODEL="gpt2", PROMPT=txt("ROMEO:"), TEMP=0.8),
        B("nb_save", MODEL="gpt2", FILE=txt("gpt2_shakespeare.pt")),
    ],
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
        B("nb_rl_watch", MODEL="driver", WORLD="world", N=3),
        say("Average score: ", rnd(B("nb_rl_score", MODEL="driver", WORLD="world", N=5), 1)),
        B("nb_save", MODEL="driver", FILE=txt("car_driver.pt")),
        B("nb_rl_play", WORLD="world"),
    ],
    comments=[comment("Things to try: make the terrain 'steps' or 'mixed', add 'can jump' or 'can rocket boost', "
                      "remove a sense and see if it still learns, change the rewards (reward speed? punish energy?), "
                      "or switch 'same world every try' off so it must handle any terrain.", 800, 40, 340, 150)],
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
        B("nb_rl_watch", MODEL="driver", WORLD="world", N=1),
    ],
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
        B("nb_rl_watch", MODEL="balancer", WORLD="world", N=2),
        B("nb_rl_play", WORLD="world"),
    ],
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
        B("nb_rl_watch", MODEL="explorer", WORLD="maze", N=1),
        B("nb_rl_play", WORLD="maze"),
    ],
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
        B("nb_rl_watch", MODEL="pilot", WORLD="world", N=3),
        B("nb_rl_play", WORLD="world"),
    ],
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
        B("nb_rl_watch", MODEL="bird", WORLD="world", N=2),
        B("nb_rl_play", WORLD="world"),
    ],
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
        B("nb_rl_watch", MODEL="driver", WORLD="world", N=2),
    ],
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
