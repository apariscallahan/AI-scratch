"""Training blocks."""
from __future__ import annotations

from .core import INDENT, ORDER_FUNCTION_CALL, BlockCtx, Setting
from .helpers import fmt_call, int_code, setting, settings_kwargs
from .spec import Drop, Num, Ref, Stack, Str, Toggle, block

TS = "TrainSetting"
RS = "RLSetting"


@block("nb_train", "training", "train %MODEL on %DATA %SETTINGS",
       MODEL=Ref("model", "model"), DATA=Ref("dataset", "data"), SETTINGS=Stack(TS), section="Train",
       prefill={"SETTINGS": [("nb_t_epochs", {"N": 10}), ("nb_t_batch", {"N": 32}),
                             ("nb_t_optimizer", {"OPT": "adam", "LR": 0.001})]},
       tooltip="Teach the model using the dataset. Put training settings inside.",
       help="Training shows the model batches of examples, measures how wrong it is (the loss) and nudges every "
            "weight to be a little less wrong. One epoch = one pass over all training examples. Watch the "
            "Charts tab: training loss should go down; if test loss goes up while training loss goes down, "
            "the model is memorising (overfitting).")
def _train(b: BlockCtx):
    kws, pre = settings_kwargs(b, "SETTINGS", context="training")
    return pre + fmt_call("nb.train", [b.ref("MODEL"), b.ref("DATA")] + kws + b.bid_kw()).split("\n")


def _ts(type_, message, tooltip, **kw):
    return block(type_, "training", message, shape="setting", check=[TS], tooltip=tooltip, **kw)


@_ts("nb_t_epochs", "for %N epochs", "How many times to go through all the training examples.",
     N=Num(10, integer=True), section="Training settings (go inside 'train')")
def _epochs(b):
    return setting("epochs", int_code(b, "N"))


@_ts("nb_t_steps", "for %N steps", "Train for an exact number of steps (batches). Language models use steps.",
     N=Num(1000, integer=True))
def _steps(b):
    return setting("steps", int_code(b, "N"))


@_ts("nb_t_batch", "batch size %N", "How many examples the model looks at before each update.",
     N=Num(32, integer=True))
def _batch(b):
    return setting("batch_size", int_code(b, "N"))


@_ts("nb_t_optimizer", "optimizer %OPT learning rate %LR",
     "The method that updates the weights, and how big each step is.",
     OPT=Drop([("Adam", "adam"), ("AdamW", "adamw"), ("SGD with momentum", "sgd"), ("RMSprop", "rmsprop")], "adam"),
     LR=Num(0.001))
def _opt(b):
    return [setting("optimizer", b.q("OPT")), setting("lr", b.val("LR"))]


@_ts("nb_t_lr", "learning rate %LR", "How big each learning step is (too big = chaos, too small = slow).",
     LR=Num(0.001))
def _lr(b):
    return setting("lr", b.val("LR"))


@_ts("nb_t_schedule", "learning rate schedule %SCHED warm up %WARM steps",
     "Change the learning rate during training: start gently (warm-up), then slowly lower it.",
     SCHED=Drop([("cosine decay", "cosine"), ("linear decay", "linear"), ("step down", "step"),
                 ("constant", "constant")], "cosine"), WARM=Num(100, integer=True))
def _sched(b):
    return [setting("schedule", b.q("SCHED")), setting("warmup", int_code(b, "WARM"))]


@_ts("nb_t_wd", "weight decay %WD", "Gently pull weights towards zero to reduce overfitting.", WD=Num(0.01))
def _wd(b):
    return setting("weight_decay", b.val("WD"))


@_ts("nb_t_loss", "loss %LOSS", "How mistakes are measured. 'auto' picks the right one for the task.",
     LOSS=Drop([("auto", "auto"), ("cross-entropy (classes)", "cross_entropy"), ("mean squared error", "mse"),
                ("mean absolute error", "mae"), ("Huber", "huber"), ("binary cross-entropy", "bce")], "auto"))
def _loss(b):
    return setting("loss", b.q("LOSS"))


@_ts("nb_t_clip", "clip gradients at %CLIP", "Limit how big a single update can be (keeps training stable).",
     CLIP=Num(1.0))
def _clip(b):
    return setting("clip", b.val("CLIP"))


@_ts("nb_t_early", "stop early after %N checks without improvement",
     "Stop training when the test loss stops getting better.", N=Num(3, integer=True))
def _early(b):
    return setting("patience", int_code(b, "N"))


@_ts("nb_t_eval", "check progress every %N steps", "How often a language model is tested on validation text.",
     N=Num(100, integer=True))
def _eval(b):
    return setting("eval_every", int_code(b, "N"))


@_ts("nb_t_goal", "goal %GOAL", "What to learn. 'rebuild the input' makes an autoencoder.",
     GOAL=Drop([("predict the answer", "auto"), ("rebuild the input (autoencoder)", "reconstruct"),
                ("clean up noisy input (denoiser)", "denoise")], "auto"))
def _goal(b):
    return setting("objective", b.q("GOAL"))


@_ts("nb_t_precision", "mixed precision %MODE", "On GPUs, compute with 16-bit numbers: faster, less memory.",
     MODE=Drop([("auto (GPU only)", "auto"), ("off", "off")], "auto"))
def _prec(b):
    return setting("precision", b.q("MODE"))


@_ts("nb_t_compile", "compile the model for speed %ON", "Use torch.compile on NVIDIA GPUs (slow start, faster "
                                                        "steps). Ignored on CPU.", ON=Toggle(True))
def _compile(b):
    return setting("compile", "True" if b.flag("ON") else "False")


@_ts("nb_t_label", "name this run %LABEL", "The name used for this run's lines in the charts.",
     LABEL=Str("run 1"))
def _label(b):
    return setting("label", f"nb.text({b.val('LABEL')})")


@_ts("nb_t_save_best", "save the best version to file %FILE", "Whenever the test score improves, save the model.",
     FILE=Str("best_model.pt"))
def _save_best(b):
    return setting("save_best", b.val("FILE"))


@block("nb_t_every", "events", "every %N %UNIT do %DO", shape="setting", check=[TS, RS],
       N=Num(100, integer=True), UNIT=Drop([("steps", "steps"), ("epochs", "epochs")], "steps"), DO=Stack(None),
       section="While training",
       tooltip="Run some blocks regularly while training — e.g. write sample text or redraw the decision map.",
       help="Goes inside a 'train' block. The blocks inside run every N steps (or epochs), so you can watch "
            "the model improve as it learns.")
def _every(b: BlockCtx):
    fn = b.c.fresh("every")
    body = b.stmts("DO")
    globs = b.c.global_names()
    lines = ["", f"def {fn}():"]
    inner = []
    if globs:
        inner.append(f"global {', '.join(sorted(set(globs)))}")
    inner.extend(body)
    if not body:
        inner.append("pass")
    lines.extend(INDENT + l for l in inner)
    lines.append("")
    return Setting("every", f"({int_code(b, 'N')}, {b.q('UNIT')}, {fn})", pre=lines)


@block("nb_metric", "training", "%WHICH of %MODEL", shape="value", output="Number",
       WHICH=Drop([("last test loss", "val_loss"), ("best test loss", "best_val_loss"),
                   ("last test accuracy", "val_acc"), ("best test accuracy", "best_val_acc"),
                   ("last training loss", "train_loss"), ("training time (s)", "train_seconds"),
                   ("steps trained", "steps")], "val_loss"),
       MODEL=Ref("model", "model"), section="Results",
       tooltip="A number from the model's training history (use it in 'say', charts or 'check that').")
def _metric(b):
    return f"nb.metric_of({b.ref('MODEL')}, {b.q('WHICH')})", ORDER_FUNCTION_CALL
