"""Output blocks: using a trained model, step by step.

A training stack ends with a 📦 weights block. The user draws a wire from it to an output start
block and builds the output stack underneath: text → tokens → scores → probabilities → a choice →
text again (or numbers, pictures, actions in a world). The blocks spell out what usually happens
hidden inside a single "predict" or "generate" call.
"""
from __future__ import annotations

from .core import ORDER_FUNCTION_CALL, BlockCtx
from .helpers import int_code
from .spec import Drop, Num, Port, Ref, Str, Val, block


def _m(b: BlockCtx, method: str, *args: str) -> str:
    """A call on the wired weights, e.g. ``gpt_weights.tokens_of(prompt)``."""
    return f"{b.wired()}.{method}({', '.join(args)})"


# ===========================================================================
# Connecting training to output
# ===========================================================================


@block("nb_weights", "output", "📦 weights of trained %MODEL %PORT", shape="terminal",
       MODEL=Ref("model", "model"), PORT=Port("out"), section="Connect training → output",
       tooltip="Put this at the very bottom of a training stack. It holds what training produced — the "
               "model's weights. Drag a wire from its ● to the ○ of a 'start output' block.",
       help="Training changes millions of numbers inside the model, called weights. This block packs them up, "
            "together with everything needed to use them: for a language model the tokenizer (how text becomes "
            "numbers) and how many tokens it can see; for a classifier the class names and how inputs were "
            "scaled. For classic models (like trees) it holds the learned rules, and for simulations the "
            "policy. Nothing can go below it: it is the end of training. Wire it to a 'start output' block to "
            "use the model.")
def _weights(b: BlockCtx):
    ident = b.c.weights_idents.get(b.id)
    if ident is None or b.c.stack_kind != "training":
        b.error("The 📦 weights block goes at the very bottom of a 'start training when ▶ clicked' stack.")
        return []
    return [f"{ident} = nb.weights({b.ref('MODEL')}{b.bid()})"]


@block("nb_when_output", "output", "%PORT start output", shape="hat", PORT=Port("in"),
       tooltip="The output stack starts here. Wire a 📦 weights block to the ○, then build what the trained "
               "model should do underneath. Output stacks run after training has finished.",
       help="Output is how a trained model gets used. Training (the yellow stack) changes the model's weights; "
            "the 📦 block at its bottom hands them over through the wire. Under this block you build the output "
            "step by step, the way real AI programs do: turn your input into numbers the model understands "
            "(tokens, or a list of numbers), run the model to get scores, turn the scores into probabilities, "
            "pick a choice, and turn the choice back into text, a name or an action. Results appear in the "
            "Output tab.")
def _when_output(b: BlockCtx):
    return []  # the compiler builds output stacks itself (it needs to know the wire first)


# ===========================================================================
# Language models: text ↔ tokens, next-token scores
# ===========================================================================


@block("nb_o_tokens", "output", "tokens of %TEXT", shape="value", output=None, TEXT=Str("Once upon a time"),
       section="Language models: text ↔ tokens",
       tooltip="Split text into tokens: the numbers a language model reads. Uses the same tokenizer as in "
               "training.",
       help="A model can't read letters, only numbers. The tokenizer (made when the text dataset was loaded) "
            "gives every character, word or word-piece a number. 'tokens of \"hi!\"' with character tokens might "
            "be [46, 47, 2]. Store the result in a variable — it's a list you can add new tokens to.")
def _tokens(b: BlockCtx):
    return _m(b, "tokens_of", b.val("TEXT")), ORDER_FUNCTION_CALL


@block("nb_o_text", "output", "text of tokens %TOKENS", shape="value", output="String", TOKENS=Val(None),
       tooltip="Turn tokens (numbers) back into text — the opposite of 'tokens of'. Works with a list or one "
               "token.")
def _text_of(b: BlockCtx):
    return _m(b, "text_of", b.val("TOKENS", default="[]")), ORDER_FUNCTION_CALL


@block("nb_o_next", "output", "scores for the next token after %TOKENS", shape="value", output=None,
       TOKENS=Val(None),
       tooltip="Run the language model once: it gives every token in its vocabulary a score for how well it "
               "would come next. Higher score = more likely.",
       help="This is the one thing a GPT really does: look at the tokens so far and score every possible next "
            "token (these raw scores are called logits). Writing a story is just this, again and again: score, "
            "pick a token, add it, repeat. The model can only look back a fixed number of tokens (its context "
            "length) — older tokens are left out automatically.")
def _next(b: BlockCtx):
    return _m(b, "next_token_scores", b.val("TOKENS", default="[]")), ORDER_FUNCTION_CALL


# ===========================================================================
# Any model
# ===========================================================================


@block("nb_o_run", "output", "run the model on %INPUT", shape="value", output=None, INPUT=Val(None),
       section="Run the model",
       tooltip="Feed one input through the trained network and get its raw output: scores (one per class or "
               "token), numbers, or a picture.",
       help="What you give it depends on what the model learned: a list of numbers for tables and toy data "
            "(e.g. [5.1, 3.5, 1.4, 0.2]), a text for a text classifier, a picture (from 'ask for a drawing') for "
            "an image model, tokens for a language model, or the senses of a simulation. Inputs are scaled the "
            "same way as the training data. Classifiers answer with one score per class — turn them into "
            "probabilities next. Number predictors answer with the number(s) directly.")
def _run(b: BlockCtx):
    return _m(b, "run", b.val("INPUT", default="None")), ORDER_FUNCTION_CALL


# ===========================================================================
# Scores → probabilities → a choice
# ===========================================================================


@block("nb_o_probs", "output", "probabilities from scores %SCORES temperature %TEMP", shape="value",
       output=None, SCORES=Val(None), TEMP=Num(1.0), section="Scores → probabilities → a choice",
       tooltip="Softmax: turn raw scores into probabilities that add up to 1. Temperature below 1 makes the "
               "favourite even more likely; above 1 evens things out (more surprising choices).",
       help="Each score is turned into e^(score ÷ temperature), then everything is divided by the total so it "
            "adds up to 1 (100%). Temperature 0 puts 100% on the highest score. For language models, low "
            "temperature gives safe, repetitive text and high temperature gives creative nonsense.")
def _probs(b: BlockCtx):
    return f"nb.softmax({b.val('SCORES', default='[]')}, temperature={b.val('TEMP')})", ORDER_FUNCTION_CALL


@block("nb_o_pick", "output", "pick %HOW choice from %PROBS", shape="value", output="Number",
       HOW=Drop([("a random", "random"), ("the most likely", "best")], "random"), PROBS=Val(None),
       tooltip="Choose one option using the probabilities. Random: a 30% option is picked 30% of the time. "
               "Most likely: always the top one. The answer is the choice's number (for text: a token).",
       help="Choices are numbered from 0, like token numbers and class numbers. Picking at random is called "
            "sampling — it's why a chatbot can answer the same question differently each time. Use 'name of "
            "choice' to see what a choice means.")
def _pick(b: BlockCtx):
    return f"nb.pick({b.val('PROBS', default='[]')}, {b.q('HOW')})", ORDER_FUNCTION_CALL


@block("nb_o_name", "output", "name of choice %CHOICE", shape="value", output="String", CHOICE=Num(0, integer=True),
       tooltip="What a choice number means: the token's text for a language model, the class name for a "
               "classifier, the action for a simulation.")
def _name(b: BlockCtx):
    return _m(b, "name_of", b.val("CHOICE")), ORDER_FUNCTION_CALL


@block("nb_o_chance", "output", "chance of choice %CHOICE in %PROBS", shape="value", output="Number",
       CHOICE=Num(0, integer=True), PROBS=Val(None),
       tooltip="The probability (0 to 1) of one choice. Choices are numbered from 0.")
def _chance(b: BlockCtx):
    return f"nb.chance_of({b.val('PROBS', default='[]')}, {b.val('CHOICE')})", ORDER_FUNCTION_CALL


@block("nb_o_show_top", "output", "show the top %N choices in %PROBS", N=Num(5, integer=True), PROBS=Val(None),
       tooltip="A bar chart (Output tab) of the most likely choices with their names. Inside a loop it updates "
               "live, so you can watch a language model make up its mind.")
def _show_top(b: BlockCtx):
    return [f"{b.wired()}.show_top({b.val('PROBS', default='[]')}, {int_code(b, 'N')}{b.bid()})"]


@block("nb_o_fact", "output", "the model's %WHAT", shape="value", output="Number",
       WHAT=Drop([("context length (tokens it can see)", "context"), ("vocabulary size (tokens it knows)", "vocab"),
                  ("number of classes", "classes"), ("number of inputs", "inputs"),
                  ("number of weights", "weights"), ("noise size (picture generators)", "noise")], "context"),
       tooltip="A fact about the wired model, e.g. how many tokens it can look back at.")
def _fact(b: BlockCtx):
    return _m(b, "fact", b.q("WHAT")), ORDER_FUNCTION_CALL


# ===========================================================================
# Talking to the program
# ===========================================================================


@block("nb_o_ask", "output", "ask %QUESTION and wait", QUESTION=Str("What should the story start with?"),
       section="Ask & show",
       tooltip="Show a question with a text box in the Output tab and wait for your answer (like Scratch's "
               "'ask and wait'). The text goes into 'answer'.")
def _ask(b: BlockCtx):
    return [f"nb.ask({b.val('QUESTION')}{b.bid()})"]


@block("nb_o_ask_draw", "output", "ask for a drawing %QUESTION and wait", QUESTION=Str("Draw a digit!"),
       tooltip="Show a drawing pad in the Output tab and wait. Your drawing goes into 'answer' as a picture you "
               "can run an image model on.")
def _ask_draw(b: BlockCtx):
    return [f"nb.ask_drawing({b.val('QUESTION')}{b.bid()})"]


@block("nb_o_answer", "output", "answer", shape="value", output=None,
       tooltip="What you typed (or drew) for the last 'ask'. Empty if nothing was asked yet.")
def _answer(b: BlockCtx):
    return "nb.answer()", ORDER_FUNCTION_CALL


@block("nb_o_show_text", "output", "show %TEXT as output", TEXT=Str("Hello!"),
       tooltip="Write text into the Output tab. Used again and again in a loop it updates the same box, so "
               "generated text appears token by token.")
def _show_text(b: BlockCtx):
    return [f"nb.show_text({b.val('TEXT')}{b.bid()})"]


# ===========================================================================
# Picture generators
# ===========================================================================


@block("nb_o_noise", "output", "random noise for the picture generator", shape="value", output=None,
       section="Picture generators",
       tooltip="A list of random numbers: the starting point a picture generator turns into a picture. "
               "Different noise = a different picture.",
       help="A VAE or GAN learns to turn a handful of random numbers (the 'noise' or 'latent code') into a "
            "picture. Change one number a little and the picture changes a little — try it!")
def _noise(b: BlockCtx):
    return _m(b, "noise"), ORDER_FUNCTION_CALL


@block("nb_o_picture", "output", "picture drawn from noise %NOISE as class %CLASS", shape="value", output=None,
       NOISE=Val(None), CLASS=Str("any"),
       tooltip="Run the picture generator: it turns the noise into a new picture. Class 'any' picks one, or ask "
               "for a class such as 7 or sneaker.")
def _picture(b: BlockCtx):
    return _m(b, "picture_from", b.val("NOISE", default="None"), b.val("CLASS")), ORDER_FUNCTION_CALL


@block("nb_o_show_pic", "output", "show picture %PIC", PIC=Val(None),
       tooltip="Show a picture in the Output tab (a drawing, a generated picture, or what an autoencoder "
               "rebuilt).")
def _show_pic(b: BlockCtx):
    return [f"nb.show_picture({b.val('PIC', default='None')}{b.bid()})"]


# ===========================================================================
# Simulations: sense → decide → act
# ===========================================================================


@block("nb_o_try", "output", "start a new try in %WORLD", WORLD=Ref("world", "world"), section="Simulations",
       tooltip="Reset the world for the model to play in, and start recording a replay.",
       help="A trained driver (or pilot, or bird) acts in a loop: look at what it senses, run the model to get "
            "scores for each action, do the best action, and repeat until the try is over.")
def _try(b: BlockCtx):
    return [f"{b.wired()}.start_try({b.ref('WORLD')}{b.bid()})"]


@block("nb_o_senses", "output", "what the model senses", shape="value", output=None,
       tooltip="The list of numbers the world gives the model right now (speed, tilt, ground radar …).")
def _senses(b: BlockCtx):
    return _m(b, "senses"), ORDER_FUNCTION_CALL


@block("nb_o_action", "output", "action from scores %SCORES", shape="value", output=None, SCORES=Val(None),
       tooltip="Turn the model's scores into an action: the highest-scoring choice, or for steering-like "
               "controls each score squashed into -1 … 1.")
def _action(b: BlockCtx):
    return _m(b, "action_from", b.val("SCORES", default="[]")), ORDER_FUNCTION_CALL


@block("nb_o_do", "output", "do action %ACTION", ACTION=Val(None),
       tooltip="Move the world forward one moment with this action.")
def _do(b: BlockCtx):
    return [_m(b, "do", b.val("ACTION", default="None"))]


@block("nb_o_over", "output", "this try is over", shape="value", output="Boolean",
       tooltip="True when the try has ended (crashed, finished, or ran out of time).")
def _over(b: BlockCtx):
    return _m(b, "try_over"), ORDER_FUNCTION_CALL


@block("nb_o_try_show", "output", "show the replay of this try",
       tooltip="Send the recorded try to the Sim tab as a replay.")
def _try_show(b: BlockCtx):
    return [f"{b.wired()}.show_try({', '.join(b.bid_kw())})"]


@block("nb_o_try_stat", "output", "%WHAT of this try", shape="value", output="Number",
       WHAT=Drop([("score (total reward)", "score"), ("distance", "distance"), ("steps", "steps")], "score"),
       tooltip="How the current try is going.")
def _try_stat(b: BlockCtx):
    return _m(b, "try_result", b.q("WHAT")), ORDER_FUNCTION_CALL

