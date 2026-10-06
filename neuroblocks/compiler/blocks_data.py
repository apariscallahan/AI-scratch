"""Data blocks: loading and preparing datasets."""
from __future__ import annotations

from .core import ORDER_FUNCTION_CALL, BlockCtx
from .helpers import assign, fmt_call, int_code, percent, setting, settings_kwargs
from .spec import Drop, Name, Num, Ref, Stack, Str, Toggle, block

DS = "DataSetting"


def _load(b: BlockCtx, fn: str, first: list[str], allowed: set | None = None):
    kws, pre = settings_kwargs(b, "SETTINGS", allowed=allowed, context="this dataset")
    args = first + [f"name={b.name()!r}"] + kws + b.bid_kw()
    return pre + assign(b.name_ident(), fmt_call(fn, args))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

TOY_KINDS = [("two spirals", "spirals"), ("two moons", "moons"), ("circles", "circles"),
             ("blobs (clusters)", "blobs"), ("XOR (four corners)", "xor"), ("checkerboard", "checkerboard"),
             ("sine wave → number", "sine"), ("curvy polynomial → number", "polynomial"),
             ("straight line → number", "linear")]


@block("nb_data_toy", "data", "make toy dataset %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"), KIND=Drop(TOY_KINDS, "spirals"), SETTINGS=Stack(DS),
       defines=("dataset", "NAME"), section="Ready-made data",
       prefill={"SETTINGS": [("nb_ds_samples", {"N": 600}), ("nb_ds_test", {"PCT": 20})]},
       tooltip="Generate a small 2-D dataset instantly (no download). Perfect for seeing how models learn.",
       help="Toy datasets are made of dots on a plane. Classification ones (spirals, moons, circles…) have "
            "coloured dots to separate; the '→ number' ones are curves to fit. Use 'show decision map' to "
            "watch what a model has learned.")
def _toy(b):
    return _load(b, "nb.data.toy", [b.q("KIND")])


@block("nb_data_table", "data", "load table %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"), KIND=Drop([("iris flowers (3 species)", "iris"), ("wines (3 vineyards)", "wine"),
                                                ("breast cancer scans (2 kinds)", "breast_cancer"),
                                                ("diabetes progression → number", "diabetes"),
                                                ("California house prices → number (download)", "california")], "iris"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Ready-made data",
       prefill={"SETTINGS": [("nb_ds_test", {"PCT": 20}), ("nb_ds_standardize", {"ON": True})]},
       tooltip="Classic small tables of measurements, each row an example.",
       help="Each row is one example (a flower, a wine, a patient…) described by a few numbers. The model "
            "learns to predict the last column: a class or a number.")
def _table(b):
    return _load(b, "nb.data.table", [b.q("KIND")])


@block("nb_data_images", "data", "load images %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"),
       KIND=Drop([("handwritten digits 8×8 (built in)", "digits"), ("MNIST digits 28×28 (download)", "mnist"),
                  ("Fashion-MNIST clothes (download)", "fashion"), ("CIFAR-10 colour photos (download)", "cifar10")],
                 "digits"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Ready-made data",
       tooltip="Picture datasets for image classification.",
       help="Digits 8×8 is tiny and built in. MNIST (handwritten digits) and Fashion-MNIST (clothes) are 70,000 "
            "28×28 grey pictures. CIFAR-10 is 60,000 colour photos of planes, cars, animals… Downloads are cached.")
def _images(b):
    return _load(b, "nb.data.images", [b.q("KIND")])


@block("nb_data_text", "data", "load text %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"),
       KIND=Drop([("Tiny Shakespeare (1 MB, download)", "shakespeare"), ("toy stories (built in)", "toy_stories"),
                  ("baby names (download)", "names"), ("Python code (built in)", "python")], "shakespeare"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Ready-made data",
       prefill={"SETTINGS": [("nb_ds_tokenizer", {"TOK": "characters"}), ("nb_ds_context", {"N": 128})]},
       tooltip="A long text for training a language model (it learns to predict the next character/word).",
       help="Language models read text one token at a time and learn to guess the next one. After training "
            "they can write new text in the same style. 'context length' is how many tokens they can look "
            "back at.")
def _text(b):
    return _load(b, "nb.data.text", [b.q("KIND")])


@block("nb_data_text_classes", "data", "load labelled texts %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"),
       KIND=Drop([("toy movie reviews: positive / negative (built in)", "toy_sentiment"),
                  ("IMDB movie reviews (Hugging Face)", "imdb"), ("AG news topics (Hugging Face)", "ag_news"),
                  ("SST-2 sentences (Hugging Face)", "sst2")], "toy_sentiment"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Ready-made data",
       prefill={"SETTINGS": [("nb_ds_tokenizer", {"TOK": "words"})]},
       tooltip="Short texts with a label each (e.g. positive/negative reviews) for text classification.")
def _text_classes(b):
    return _load(b, "nb.data.text_classes", [b.q("KIND")])


@block("nb_data_series", "data", "make time series %NAME : %KIND %SETTINGS",
       NAME=Name("dataset", "data"),
       KIND=Drop([("sine wave", "sine"), ("seasonal pattern", "seasonal"), ("random walk", "random_walk"),
                  ("square wave", "square")], "sine"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Ready-made data",
       prefill={"SETTINGS": [("nb_ds_window", {"W": 32, "H": 1}), ("nb_ds_noise", {"NOISE": 0.05})]},
       tooltip="A signal over time, cut into windows: look at the past, predict what comes next.")
def _series(b):
    return _load(b, "nb.data.series", [b.q("KIND")])


@block("nb_data_csv", "data", "load CSV file %FILE as %NAME predict column %TARGET %SETTINGS",
       FILE=Str("my_data.csv"), NAME=Name("dataset", "data"), TARGET=Str(""), SETTINGS=Stack(DS),
       defines=("dataset", "NAME"), section="Your own data",
       prefill={"SETTINGS": [("nb_ds_test", {"PCT": 20})]},
       tooltip="Your own spreadsheet (CSV). Leave the column empty to predict the last column.",
       help="Upload a CSV from the Data tab (it goes into the 'data' folder). Words in a column are turned into "
            "numbers automatically, missing values are filled in, and the task (classes or numbers) is "
            "guessed from the column you predict.")
def _csv(b):
    return _load(b, "nb.data.csv", [b.val("FILE"), f"target={b.val('TARGET')}"])


@block("nb_data_textfile", "data", "load text file %FILE as %NAME %SETTINGS",
       FILE=Str("my_text.txt"), NAME=Name("dataset", "data"), SETTINGS=Stack(DS),
       defines=("dataset", "NAME"), section="Your own data",
       prefill={"SETTINGS": [("nb_ds_tokenizer", {"TOK": "characters"}), ("nb_ds_context", {"N": 128})]},
       tooltip="Train a language model on your own .txt file (stories, song lyrics you wrote, code …).")
def _textfile(b):
    return _load(b, "nb.data.text_file", [b.val("FILE")])


@block("nb_data_url", "data", "load text from web address %URL as %NAME %SETTINGS",
       URL=Str("https://www.gutenberg.org/cache/epub/11/pg11.txt"), NAME=Name("dataset", "data"),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Your own data",
       prefill={"SETTINGS": [("nb_ds_tokenizer", {"TOK": "characters"}), ("nb_ds_context", {"N": 128})]},
       tooltip="Download a plain-text page (e.g. a public-domain book) and use it as a language-model dataset.")
def _url(b):
    return _load(b, "nb.data.text_url", [b.val("URL")])


@block("nb_data_folder", "data", "load image folder %FOLDER as %NAME %SETTINGS",
       FOLDER=Str("my_pictures"), NAME=Name("dataset", "data"), SETTINGS=Stack(DS),
       defines=("dataset", "NAME"), section="Your own data",
       prefill={"SETTINGS": [("nb_ds_image_size", {"N": 64}), ("nb_ds_test", {"PCT": 20})]},
       tooltip="Your own pictures: one sub-folder per class (my_pictures/cats, my_pictures/dogs …).")
def _folder(b):
    return _load(b, "nb.data.image_folder", [b.val("FOLDER")])


@block("nb_data_hf", "data", "load Hugging Face dataset %ID as %NAME\ntext column %TEXT label column %LABEL "
                             "%SETTINGS",
       ID=Str("roneneldan/TinyStories"), NAME=Name("dataset", "data"), TEXT=Str("text"), LABEL=Str(""),
       SETTINGS=Stack(DS), defines=("dataset", "NAME"), section="Your own data",
       prefill={"SETTINGS": [("nb_ds_samples", {"N": 20000}), ("nb_ds_context", {"N": 128})]},
       tooltip="Any dataset from huggingface.co (needs: pip install datasets). With a label column it becomes "
               "text classification; without, a language-model text.")
def _hf(b):
    return _load(b, "nb.data.huggingface", [b.val("ID"), f"text_column={b.val('TEXT')}",
                                            f"label_column={b.val('LABEL')}"])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@block("nb_ds_test", "data", "keep %PCT % aside for testing", shape="setting", check=DS, PCT=Num(20),
       section="Settings (go inside a dataset)",
       tooltip="Hold back some examples the model never trains on, to test it fairly.")
def _ds_test(b):
    return setting("test", percent(b, "PCT"))


@block("nb_ds_samples", "data", "use %N examples", shape="setting", check=DS, N=Num(500, integer=True),
       tooltip="How many examples to make (toy data) or to use (others — fewer is faster).")
def _ds_samples(b):
    return setting("samples", int_code(b, "N"))


@block("nb_ds_noise", "data", "noise %NOISE", shape="setting", check=DS, NOISE=Num(0.1),
       tooltip="How messy the toy data is (0 = perfectly clean).")
def _ds_noise(b):
    return setting("noise", b.val("NOISE"))


@block("nb_ds_classes", "data", "number of classes %N", shape="setting", check=DS, N=Num(3, integer=True),
       tooltip="How many groups/colours (spirals and blobs).")
def _ds_classes(b):
    return setting("classes", int_code(b, "N"))


@block("nb_ds_standardize", "data", "standardize numbers %ON", shape="setting", check=DS, ON=Toggle(True),
       tooltip="Rescale every column to average 0 and spread 1 — helps most models learn.")
def _ds_std(b):
    return setting("standardize", "True" if b.flag("ON") else "False")


@block("nb_ds_tokenizer", "data", "split text into %TOK", shape="setting", check=DS,
       TOK=Drop([("characters", "characters"), ("words", "words"), ("GPT-2 sub-words (needs tiktoken)", "gpt2")],
                "characters"),
       tooltip="Tokens are the pieces a language model reads: single characters, whole words, or GPT-2's "
               "sub-word pieces.")
def _ds_tok(b):
    return setting("tokenizer", b.q("TOK"))


@block("nb_ds_context", "data", "context length %N tokens", shape="setting", check=DS, N=Num(128, integer=True),
       tooltip="How many tokens the model sees at once. Longer = more memory and slower, but better long-range "
               "understanding.")
def _ds_ctx(b):
    return setting("context", int_code(b, "N"))


@block("nb_ds_vocab", "data", "keep the %N most common words", shape="setting", check=DS,
       N=Num(5000, integer=True), tooltip="Vocabulary size for word tokens; rarer words become <unk>.")
def _ds_vocab(b):
    return setting("vocab", int_code(b, "N"))


@block("nb_ds_lowercase", "data", "make text lowercase %ON", shape="setting", check=DS, ON=Toggle(True),
       tooltip="Treat 'The' and 'the' as the same.")
def _ds_lower(b):
    return setting("lowercase", "True" if b.flag("ON") else "False")


@block("nb_ds_image_size", "data", "resize images to %N pixels", shape="setting", check=DS,
       N=Num(32, integer=True), tooltip="Make every image N×N pixels (smaller = faster).")
def _ds_size(b):
    return setting("size", int_code(b, "N"))


@block("nb_ds_grayscale", "data", "make images grey %ON", shape="setting", check=DS, ON=Toggle(True),
       tooltip="Drop colour (3 channels → 1).")
def _ds_gray(b):
    return setting("grayscale", "True" if b.flag("ON") else "False")


@block("nb_ds_augment", "data", "augment images with %AUG", shape="setting", check=DS,
       AUG=Drop([("small shifts", "shift"), ("mirror flips", "flip"), ("flips + shifts", "flip+shift"),
                 ("shifts + rotations", "shift+rotate")], "shift"),
       tooltip="Randomly change training images a little each time, so the model can't just memorise them.")
def _ds_aug(b):
    return setting("augment", b.q("AUG"))


@block("nb_ds_window", "data", "look at %W past values, predict %H next", shape="setting", check=DS,
       W=Num(32, integer=True), H=Num(1, integer=True), tooltip="Window sizes for time series.")
def _ds_window(b):
    return [setting("window", int_code(b, "W")), setting("horizon", int_code(b, "H"))]


@block("nb_ds_task", "data", "task %TASK", shape="setting", check=DS,
       TASK=Drop([("guess automatically", "auto"), ("choose a class", "classification"),
                  ("predict a number", "regression")], "auto"),
       tooltip="For CSV files: is the column you predict a category or a number?")
def _ds_task(b):
    return setting("task", b.q("TASK"))


@block("nb_ds_seed", "data", "shuffle with seed %SEED", shape="setting", check=DS, SEED=Num(0, integer=True),
       tooltip="Change how examples are shuffled and split (same seed = same split).")
def _ds_seed(b):
    return setting("seed", int_code(b, "SEED"))


# ---------------------------------------------------------------------------
# Looking at data
# ---------------------------------------------------------------------------


@block("nb_data_plot", "data", "plot %DATA as dots", DATA=Ref("dataset", "data"), section="Look at data",
       tooltip="Scatter plot of the dataset (big tables are squashed to 2-D with PCA).")
def _plot(b):
    return [f"nb.plot_data({b.ref('DATA')}{b.bid()})"]


@block("nb_data_examples", "data", "show %N examples from %DATA", N=Num(16, integer=True),
       DATA=Ref("dataset", "data"), tooltip="Show some examples (pictures, rows or text) in the Results tab.")
def _examples(b):
    return [f"nb.show_examples({b.ref('DATA')}, {int_code(b, 'N')}{b.bid()})"]


@block("nb_data_map", "data", "draw a map of %DATA using %METHOD", DATA=Ref("dataset", "data"),
       METHOD=Drop([("PCA (fast)", "pca"), ("t-SNE (slower, finds clusters)", "tsne")], "pca"),
       tooltip="Squash many numbers per example into a 2-D map where similar examples land close together.")
def _map(b):
    return [f"nb.show_map({b.ref('DATA')}, {b.q('METHOD')}{b.bid()})"]


@block("nb_data_count", "data", "number of examples in %DATA", shape="value", output="Number",
       DATA=Ref("dataset", "data"), tooltip="How many examples the dataset has (training + test).")
def _count(b):
    return f"nb.data.size({b.ref('DATA')})", ORDER_FUNCTION_CALL


@block("nb_data_nclasses", "data", "number of classes in %DATA", shape="value", output="Number",
       DATA=Ref("dataset", "data"), tooltip="How many different answers (classes) the dataset has.")
def _nclasses(b):
    return f"nb.data.class_count({b.ref('DATA')})", ORDER_FUNCTION_CALL
