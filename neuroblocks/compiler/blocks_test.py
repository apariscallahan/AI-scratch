"""Test blocks, classic ML blocks and file blocks."""
from __future__ import annotations

from .core import ORDER_FUNCTION_CALL, BlockCtx
from .helpers import assign, fmt_call, int_code
from .spec import Drop, Name, Num, Ref, Str, Val, block

LEGACY = ("This is an old shortcut block that hides how a model's output is made. The new way: end the training "
          "stack with '📦 weights of trained …', wire it to '○ start output' (Output blocks), and build the "
          "output step by step under it.")

# ===========================================================================
# Test
# ===========================================================================


@block("nb_evaluate", "testing", "test %MODEL on %DATA", MODEL=Ref("model", "model"), DATA=Ref("dataset", "data"),
       section="Test",
       tooltip="Score the model on the test examples it never trained on: accuracy, errors, confusion matrix.")
def _evaluate(b):
    return [f"nb.evaluate({b.ref('MODEL')}, {b.ref('DATA')}{b.bid()})"]


@block("nb_score", "testing", "%METRIC of %MODEL on %DATA", shape="value", output="Number",
       METRIC=Drop([("accuracy", "accuracy"), ("average error", "mae"), ("R² score", "r2"), ("loss", "loss"),
                    ("F1 score", "f1"), ("perplexity", "perplexity")], "accuracy"),
       MODEL=Ref("model", "model"), DATA=Ref("dataset", "data"),
       tooltip="A test score as a number (accuracy is between 0 and 1).")
def _score(b):
    return f"nb.score({b.ref('MODEL')}, {b.ref('DATA')}, {b.q('METRIC')})", ORDER_FUNCTION_CALL


@block("nb_check", "testing", "check that %VALUE %OP %TARGET", VALUE=Val(None),
       OP=Drop([("≥", ">="), (">", ">"), ("≤", "<="), ("<", "<"), ("=", "=="), ("≠", "!=")], ">="),
       TARGET=Num(0.9),
       tooltip="A test: passes (✅) or fails (❌). On the command line, failed checks make the run exit with an "
               "error — useful for automatic testing.")
def _check(b):
    return [f"nb.check({b.val('VALUE', default='None')}, {b.q('OP')}, {b.val('TARGET')})"]


@block("nb_show_predictions", "testing", "show predictions of %MODEL on %N examples from %DATA",
       MODEL=Ref("model", "model"), N=Num(16, integer=True), DATA=Ref("dataset", "data"),
       tooltip="See what the model guesses for some test examples (green = right, red = wrong).")
def _preds(b):
    return [f"nb.show_predictions({b.ref('MODEL')}, {b.ref('DATA')}, {int_code(b, 'N')}{b.bid()})"]


@block("nb_show_confusion", "testing", "show confusion matrix of %MODEL on %DATA", MODEL=Ref("model", "model"),
       DATA=Ref("dataset", "data"), tooltip="A grid showing which classes get mixed up with which.")
def _confusion(b):
    return [f"nb.show_confusion({b.ref('MODEL')}, {b.ref('DATA')}{b.bid()})"]


@block("nb_decision_map", "testing", "show decision map of %MODEL on %DATA", MODEL=Ref("model", "model"),
       DATA=Ref("dataset", "data"),
       tooltip="Colour the whole plane by the model's prediction (for 2-D toy data). Put it in 'every N steps' "
               "to watch the boundary form.")
def _dmap(b):
    return [f"nb.show_decision_map({b.ref('MODEL')}, {b.ref('DATA')}{b.bid()})"]


@block("nb_predict", "testing", "prediction of %MODEL for %INPUT", shape="value", output=None,
       MODEL=Ref("model", "model"), INPUT=Val(None), toolbox=False,
       tooltip="(Old shortcut) Ask the model about one example. Build an output stack instead.")
def _predict(b):
    b.warn(LEGACY)
    return f"nb.predict({b.ref('MODEL')}, {b.val('INPUT', default='None')})", ORDER_FUNCTION_CALL


@block("nb_generate", "testing", "preview: %MODEL writes %LENGTH tokens after %PROMPT creativity %TEMP",
       LENGTH=Num(150, integer=True), MODEL=Ref("model", "model"), PROMPT=Str("ROMEO:"), TEMP=Num(0.8),
       section="Peek while training",
       tooltip="A quick look at what a language model writes so far — put it in 'every N steps' to watch it "
               "improve while it trains. (To really use the model, build an output stack.)",
       help="This does the whole writing loop in one go, so you can see progress during training. The output "
            "stack shows what happens inside: tokens → scores for the next token → probabilities → pick one → "
            "repeat. Creativity is the temperature: 0 = always the most likely token, 1 = adventurous, "
            "2 = chaotic.")
def _generate(b):
    return [fmt_call("nb.generate", [b.ref("MODEL"), b.val("PROMPT"), int_code(b, "LENGTH"),
                                     f"temperature={b.val('TEMP')}"] + b.bid_kw())]


@block("nb_generated", "testing", "text written by %MODEL starting with %PROMPT length %LENGTH", shape="value",
       output="String", MODEL=Ref("model", "model"), PROMPT=Str("Once upon a time"), LENGTH=Num(100, integer=True),
       toolbox=False, tooltip="(Old shortcut) The text a language model writes. Build an output stack instead.")
def _generated(b):
    b.warn(LEGACY)
    return (f"nb.generated_text({b.ref('MODEL')}, {b.val('PROMPT')}, {int_code(b, 'LENGTH')})",
            ORDER_FUNCTION_CALL)


@block("nb_gen_show", "testing", "preview: show %N new pictures from %MODEL of class %CLASS",
       N=Num(16, integer=True), MODEL=Ref("model", "artist"), CLASS=Str("any"),
       tooltip="A quick look at the pictures a generator invents so far (class 'any', or a class name such as 7 "
               "or sneaker). To really use it, build an output stack: noise → picture.")
def _gen_show(b):
    return [f"nb.show_generated({b.ref('MODEL')}, {int_code(b, 'N')}, {b.val('CLASS')}{b.bid()})"]


@block("nb_play_chat", "testing", "let me prompt %MODEL", MODEL=Ref("model", "model"),
       toolbox=False, tooltip="(Old shortcut) Opens a prompt box so you can type and see what the model writes.")
def _chat(b):
    b.warn(LEGACY)
    return [f"nb.play_chat({b.ref('MODEL')}{b.bid()})"]


@block("nb_play_draw", "testing", "let me draw for %MODEL", MODEL=Ref("model", "model"),
       toolbox=False, tooltip="(Old shortcut) Opens a drawing pad: draw a digit and see the model's guesses live.")
def _draw(b):
    b.warn(LEGACY)
    return [f"nb.play_draw({b.ref('MODEL')}{b.bid()})"]


@block("nb_play_form", "testing", "let me type inputs for %MODEL", MODEL=Ref("model", "model"),
       toolbox=False, tooltip="(Old shortcut) Opens a form: type values for each input and get a prediction.")
def _form(b):
    b.warn(LEGACY)
    return [f"nb.play_form({b.ref('MODEL')}{b.bid()})"]


# ===========================================================================
# Classic ML
# ===========================================================================


def _classic(b: BlockCtx, fn: str, *args: str):
    return assign(b.name_ident(), fmt_call(f"nb.classic.{fn}", [repr(b.name())] + list(args) + b.bid_kw()))


@block("nb_c_tree", "classic", "create decision tree %NAME max depth %DEPTH", NAME=Name("model", "model"),
       DEPTH=Num(4, integer=True), defines=("model", "NAME"), section="Supervised (learn from answers)",
       tooltip="A flowchart of yes/no questions learned from data. Depth = how many questions in a row "
               "(0 = no limit).")
def _tree(b):
    return _classic(b, "decision_tree", f"depth={int_code(b, 'DEPTH')}")


@block("nb_c_forest", "classic", "create random forest %NAME with %TREES trees max depth %DEPTH",
       NAME=Name("model", "model"), TREES=Num(100, integer=True), DEPTH=Num(0, integer=True),
       defines=("model", "NAME"), tooltip="Many decision trees vote together. Strong and hard to break.")
def _forest(b):
    return _classic(b, "random_forest", f"trees={int_code(b, 'TREES')}", f"depth={int_code(b, 'DEPTH')}")


@block("nb_c_boost", "classic", "create gradient boosting %NAME with %TREES trees learning rate %LR",
       NAME=Name("model", "model"), TREES=Num(100, integer=True), LR=Num(0.1), defines=("model", "NAME"),
       tooltip="Trees added one at a time, each fixing the previous ones' mistakes. Often the best on tables.")
def _boost(b):
    return _classic(b, "gradient_boosting", f"trees={int_code(b, 'TREES')}", f"lr={b.val('LR')}")


@block("nb_c_knn", "classic", "create k-nearest neighbours %NAME k = %K", NAME=Name("model", "model"),
       K=Num(5, integer=True), defines=("model", "NAME"),
       tooltip="Predict by looking at the k most similar training examples. No real 'training' needed.")
def _knn(b):
    return _classic(b, "knn", f"k={int_code(b, 'K')}")


@block("nb_c_linear", "classic", "create linear model %NAME", NAME=Name("model", "model"),
       defines=("model", "NAME"),
       tooltip="Draws the best straight line (or flat surface). For classes this is logistic regression.")
def _linear(b):
    return _classic(b, "linear_regression")


@block("nb_c_logistic", "classic", "create logistic regression %NAME", NAME=Name("model", "model"),
       defines=("model", "NAME"), tooltip="A linear model that outputs class probabilities.")
def _logistic(b):
    return _classic(b, "logistic_regression")


@block("nb_c_svm", "classic", "create support vector machine %NAME kernel %KERNEL", NAME=Name("model", "model"),
       KERNEL=Drop([("curvy (RBF)", "rbf"), ("straight (linear)", "linear"), ("polynomial", "poly")], "rbf"),
       defines=("model", "NAME"), tooltip="Finds the widest possible gap between classes.")
def _svm(b):
    return _classic(b, "svm", f"kernel={b.q('KERNEL')}")


@block("nb_c_bayes", "classic", "create naive Bayes %NAME", NAME=Name("model", "model"), defines=("model", "NAME"),
       tooltip="Simple probability-based classifier. Very fast.")
def _bayes(b):
    return _classic(b, "naive_bayes")


@block("nb_c_kmeans", "classic", "create k-means clustering %NAME with %K groups", NAME=Name("model", "model"),
       K=Num(3, integer=True), defines=("model", "NAME"), section="Unsupervised (find groups)",
       tooltip="Finds groups in data without being told the answers.")
def _kmeans(b):
    return _classic(b, "kmeans", f"k={int_code(b, 'K')}")


@block("nb_c_importance", "classic", "show which inputs matter to %MODEL", MODEL=Ref("model", "model"),
       section="Look inside", tooltip="Bar chart of feature importance (trees, forests, boosting, linear models).")
def _importance(b):
    return [f"nb.classic.show_importance({b.ref('MODEL')}{b.bid()})"]


@block("nb_c_show_tree", "classic", "show the questions in decision tree %MODEL", MODEL=Ref("model", "model"),
       tooltip="Print the learned yes/no questions of a decision tree.")
def _show_tree(b):
    return [f"nb.classic.show_tree({b.ref('MODEL')}{b.bid()})"]


# ===========================================================================
# Files
# ===========================================================================


@block("nb_save", "files", "save %MODEL to file %FILE", MODEL=Ref("model", "model"), FILE=Str("my_model.pt"),
       tooltip="Save a trained model (into the 'models' folder) so you can load it later — e.g. train on a cloud "
               "GPU, then test it on your laptop.")
def _save(b):
    return [f"nb.save({b.ref('MODEL')}, {b.val('FILE')}{b.bid()})"]


@block("nb_load", "files", "load model %NAME from file %FILE", NAME=Name("model", "model"), FILE=Str("my_model.pt"),
       defines=("model", "NAME"), tooltip="Load a model saved earlier (from the 'models' folder).")
def _load(b):
    return assign(b.name_ident(), fmt_call("nb.load_model", [b.val("FILE"), f"name={b.name()!r}"] + b.bid_kw()))
