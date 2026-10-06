# NeuroBlocks — design & brainstorm

NeuroBlocks is a Scratch-style visual block language for building, training and
testing AI / machine-learning models. You snap blocks together ("load dataset →
create network → train → test"), press the green flag, and watch it learn — and
the exact same program can be exported and run headless on a Linux cloud GPU.

This document records the brainstorm (options considered, the functionality we
need) and the architecture that was chosen.

---

## 1. How should it be built? (options considered)

| Option | Verdict |
|---|---|
| **Desktop GUI in Python (Qt/Tk) with a hand-written block editor** | Rejected. A good drag-and-drop block editor is a huge project on its own and would look worse than existing web editors. |
| **Electron app** | Rejected for now. Adds a Node toolchain and packaging for little gain; a local web page gives the same UX. |
| **Fork of Scratch 3 (scratch-gui/scratch-vm)** | Rejected. Scratch's VM runs in the browser in JavaScript, has sprite/stage assumptions everywhere, and cannot drive PyTorch or a GPU. |
| **Local web app: Blockly editor in the browser + Python backend** | **Chosen.** Blockly is the library Scratch's own editor is built on, and its *zelos* renderer draws Scratch-3-style blocks. The browser does what browsers are good at (drag-and-drop, charts, canvas animation); Python does what it is good at (PyTorch, scikit-learn, physics). Runs locally on Windows with one click, and the same server can run on a cloud box. |

### Key architecture decisions

1. **Blocks compile to real, readable Python.** The block program (Blockly's JSON)
   is compiled *in Python* into an ordinary script that calls a small runtime
   library (`import neuroblocks as nb`). Because the compiler is Python, the CLI on a
   Linux GPU box needs no browser or Node — `neuroblocks run project.nblk` works
   anywhere Python + PyTorch work. The generated code is also shown live in a
   "Python" tab: a bridge from blocks to real code, just like Scratch → Python.
2. **One source of truth for blocks.** Each block is declared once in Python
   (`neuroblocks/compiler/blocks_*.py`): its look (Blockly JSON), toolbox entry,
   help text and code generator. The browser fetches the definitions from the
   server, so the editor, compiler, CLI and docs can never drift apart.
3. **Programs run in a subprocess.** Training never freezes the UI, Stop always
   works, GPU memory is released, and a crash can't take down the editor. The
   runtime streams structured *events* (metrics, logs, images, simulation frames,
   which block is running…) as JSON lines; the server relays them to the browser
   over a WebSocket. On the CLI, the same events are pretty-printed and saved to
   `events.jsonl`, which the GUI can open later ("run in the cloud, look at it at
   home").
4. **Containers + settings blocks, not giant blocks.** Big blocks (train, dataset,
   world) are C-shaped containers holding small optional *setting* blocks
   (`for 5 epochs`, `optimizer Adam lr 0.001`, `sense: ground ahead`). Blockly's
   connection *checks* make a setting only snap into containers where it makes
   sense — the puzzle-piece shapes teach what goes where.
5. **Numbers are value slots**, like Scratch: any numeric setting can take a
   variable or math expression, so loops/hyperparameter sweeps just work.
6. **Named things.** Datasets, models and worlds have names typed into their
   "create" block; other blocks pick them from dropdowns that update live (and
   renames propagate). Default names (`data`, `model`, `world`) mean beginners
   never have to think about it.
7. **Training and output are separate stacks, joined by a wire.** A training stack ends with a
   cap-shaped *📦 weights of trained …* block; an output stack starts with *○ start output*. The
   user drags a wire (node-editor style) from the weights block's ● to the ○. The output stack
   spells out inference step by step — tokens, next-token scores, softmax with temperature,
   sampling, decoding — instead of hiding it in one "predict"/"generate" block, so learners see
   how a model's output is really made. The wire is stored as the in-port's field value (the
   weights block's id), so saving, undo/redo and copy/paste need nothing special; the editor only
   draws it. Output stacks run after every training stack and compile to calls on a `Weights`
   object (`gpt_weights.next_token_scores(tokens)`, `nb.softmax(…)`, `nb.pick(…)`).
8. **Smart defaults and auto-fixes.** Output layers size themselves from the data,
   input shapes are inferred, a missing Flatten is inserted with a friendly
   warning, the loss function is chosen from the task, the device is picked
   automatically (CUDA → Apple MPS → CPU).

---

## 2. Functionality brainstorm

### 2.1 The editor (visual element)
- Scratch-style blocks (zelos renderer), colour-coded categories, a green-flag *start training* hat
  block and a *start output* hat block, joined by node-editor style wires.
- Toolbox with labelled sub-sections; containers come pre-filled with sensible settings.
- The block that is currently running glows; a failing block turns red with the error attached.
- Static checks before running (missing names, settings in the wrong place).
- Right-hand panel tabs: **Console** (say/log/errors/tests), **Charts** (live
  loss/accuracy/reward curves), **Results** (decision boundaries, prediction grids,
  confusion matrices, generated text), **Sim** (animated physics replays), **Model**
  (layer-by-layer summary with shapes & parameter counts), **Data** (dataset
  previews), **Output** (what output stacks make, and interactive sessions), **Python**
  (generated code).
- Projects: save/open (`.nblk` JSON), autosave, examples gallery, undo/redo,
  zoom, block help.
- Run controls: ▶ run, ⏭ "skip" (finish the current training early and carry on),
  ⏹ stop, *quick test* mode (shrinks every training loop to a few steps).

### 2.2 Data
- Built-in toy data generated on the fly (moons, circles, spirals, blobs, XOR,
  checkerboard, sine wave, polynomial) — instant, offline, great for intuition.
- Classic tabular sets bundled with scikit-learn (iris, wine, breast cancer,
  diabetes); downloads for others.
- Images: digits 8×8 (offline), MNIST, Fashion-MNIST, CIFAR-10 (downloaded & cached),
  your own folder of images.
- Text: Tiny Shakespeare, baby names, a built-in offline "toy stories" corpus, your own
  text file or URL, Hugging Face datasets (optional).
- CSV files with automatic handling of categorical columns, missing values and task detection.
- Settings: test split, standardise, subset, noise/samples, tokeniser (characters /
  words / GPT-2 BPE), context length, image resize, augmentation, target column.

### 2.3 Models
- *Create neural network* container + layer blocks: dense, output (auto-sized),
  activation, dropout, batch/layer norm, flatten, conv2D, pooling, global pool,
  upsample, token & positional embedding, transformer block (multi-head attention),
  LSTM/GRU, plus structural blocks **repeat N times { … }** and **residual { … }**.
- Prebuilt: GPT language model (layers/heads/width/dropout), simple MLP.
- Pretrained Hugging Face language models for fine-tuning (optional, best on a GPU).
- Classic ML (scikit-learn): linear/logistic regression, k-NN, decision tree,
  random forest, gradient boosting, SVM, naive Bayes, k-means, PCA.

### 2.4 Training
- One *train* block works for neural nets, language models and classic models.
- Settings: epochs or steps, batch size, optimiser (Adam/AdamW/SGD/RMSprop) and
  learning rate, LR schedule (cosine/linear/step + warm-up), weight decay, loss,
  gradient clipping, early stopping, mixed precision, objective (predict /
  reconstruct-as-autoencoder), run label, save-best checkpoint.
- Event blocks inside training: **every N steps do { … }** (e.g. generate sample text
  while an LLM trains, redraw the decision boundary).
- Live metrics, ETA, throughput; graceful "skip".

### 2.5 Testing & evaluation
- Evaluate (accuracy, precision/recall, confusion matrix, R², MAE, perplexity).
- Show predictions on examples, decision boundaries, reconstructions.
- Reporter blocks (accuracy of…, last loss of…, prediction for…, generated text…)
  so results can drive logic, loops and charts.
- **check that … ≥ …** test assertions → ✅/❌ summary and a non-zero exit code on the CLI (CI-friendly).
- Output stacks (see §2.9) and an **Output** tab: text written token by token, live
  top-choice charts, pictures, and *ask … and wait* boxes (text, or a drawing pad).

### 2.6 Simulations & reinforcement learning
- Worlds: **car on terrain** (pymunk 2D physics), cart-pole, mountain car, pendulum,
  rocket lander, maze/grid world, flappy bird.
- The car world is fully buildable from blocks: terrain (flat/hills/rough/steps,
  bumpiness, length, same-or-new each episode), car body, **abilities** (drive,
  brake, lean, jump, rocket boost), **sensors** (speed, tilt, spin, ground radar
  rays, wheel contact, height, progress, fuel), **rewards/penalties** (distance,
  speed, upright, finish bonus, flip, energy, time, custom expressions) and **episode
  end conditions**.
- Learning algorithms: PPO (general purpose), DQN, neuro-evolution (a population of
  cars learning together — very visual) and tabular Q-learning for mazes. The
  policy network is built with the same layer blocks as any other model.
- Watch replays during and after training, design worlds before training ("show world").

### 2.7 Output & general programming
- say, charts from lists, custom live chart points, sounds, tables.
- Control (repeat, for-each, if/else, while, wait, stop), operators, variables,
  lists, functions ("My Blocks"), hyperparameter blocks overridable from the CLI.
- Save/load models (weights + tokenizer + class names + normalisation), so a model
  trained in the cloud can be loaded locally for testing.

### 2.8 Cloud GPU / CLI
- `neuroblocks run project.nblk --device cuda` — headless run, pretty progress,
  `events.jsonl`, saved models; `--set name=value` overrides hyperparameters;
  `--quick` smoke test.
- `neuroblocks export project.nblk -o train.py` — standalone Python script.
- `neuroblocks bundle project.nblk -o job.zip` — everything needed on a fresh Linux
  GPU machine (script, project, runtime package, data files, requirements,
  `run.sh`, Dockerfile, README). `unzip job.zip && bash run.sh`.
- `neuroblocks serve --host 0.0.0.0 --token …` on the GPU box + "Run on: remote" in
  the GUI → design locally, train remotely with live charts (via SSH tunnel or token).
- `neuroblocks view runs/<run>` — open a finished (cloud) run's charts and replays in the GUI.

### 2.9 Output stacks (using a trained model)
- *📦 weights of trained [model]* — the cap at the bottom of a training stack; packs the weights
  with the tokenizer / class names / input scaling / policy needed to use them.
- *○ start output* — wired to a weights block; runs after training.
- Language models: *tokens of*, *text of tokens*, *scores for the next token after*.
- Any model: *run the model on* (numbers, text, a drawing, tokens, a world's senses).
- *probabilities from scores … temperature …*, *pick a random / the most likely choice*,
  *name of choice*, *chance of choice*, *show the top N choices* (live bar chart).
- Picture generators: *random noise*, *picture drawn from noise … as class …*, *show picture*.
- Simulations: *start a new try in*, *what the model senses*, *action from scores*,
  *do action*, *this try is over*, *show the replay of this try*.
- *ask … and wait* / *ask for a drawing … and wait* / *answer*, *show … as output*.
- The old one-block shortcuts (*prediction of*, *let me prompt / draw / type*) still load and run
  in old projects but are hidden from the toolbox; *preview: … writes N tokens* stays as a way
  to peek at a language model while it trains.

---

## 3. Architecture

```
 Browser (Blockly + panels)  ──HTTP/WebSocket──  Python server (FastAPI)
        │  workspace JSON                              │ compile (Python)
        ▼                                              ▼
   generated Python preview                 subprocess: python script.py
                                                       │ events (JSON lines)
                                                       ▼
                                     neuroblocks.runtime (PyTorch, sklearn, pymunk)
```

Package layout:

```
neuroblocks/
  compiler/   block DSL, block definitions + code generators, toolbox, compiler
  runtime/    the `nb` library used by generated code (data, layers, training,
              evaluation, generation, classic ML, RL worlds & algorithms, events)
  server/     FastAPI app, run manager, remote proxy, bundle export
  static/     the web editor (Blockly, Chart.js, panels, simulation renderer)
  examples/   example projects (.nblk)
  cli.py      `neuroblocks` command
```

Event protocol (runtime → GUI), one JSON object per line, e.g.
`{"type":"metric","chart":"loss","series":"model · train","x":120,"y":1.93}`.
Types: `start, device, log, say, block, progress, metric, dataset, model, image,
table, text, text_stream, confusion, scatter, plot, predictions, check,
check_summary, sim_replay, sim_live, interactive, interactive_result,
interactive_end, file, sound, train_done, weights, output_start, output_end,
output_text, bars, output_picture, ask, ask_done, error, done`.

Control messages (GUI → runtime) travel over the worker's stdin as JSON lines:
`skip`, `interact` / `interact_end` (Output-tab sessions and answers to *ask* blocks) and `keys`
(human play).

---

## 4. Implementation notes (things learned while building it)

* **Instant Run.** Importing PyTorch — and the compiler stack its optimizers import
  lazily — takes ~10 s on a laptop. The server keeps one *pre-warmed worker*
  process with everything imported, waiting for the next program on stdin, and
  starts a fresh one after each run. Pressing Run starts training in ~1 s.
* **No blocking reads on stdin.** On Windows, a thread blocked reading the stdin
  pipe deadlocks any other thread that queries the handle (many libraries call
  `isatty()` while importing). The runtime polls with `PeekNamedPipe`/`select`.
* **Lazy `nb` namespace done carefully.** `import neuroblocks as nb` resolves names
  lazily (fast startup), through an explicit name map — submodules such as
  `runtime/evaluate.py` would otherwise shadow functions like `nb.evaluate`, and
  `from . import paths` must not import the runtime as a side effect.
* **Names, not variables.** Datasets, models and worlds are named in their
  *create* block; references are dynamic dropdowns that rename automatically.
* **C-blocks like Scratch.** A container's label sits on its own row so the
  settings mouth opens underneath instead of being indented by the label width.
* **Simulations (`runtime/rl/`).** Worlds are assembled from part blocks (terrain,
  body, abilities = actions, senses = observations, rewards, endings); one base
  class assembles rewards/endings so custom reward formulas work in every world.
  The car uses pymunk (60 Hz, suspension via groove joints + springs, motorised
  wheels, a crash = the body touching the ground) and is deterministic per seed.
  The user's layer blocks become the policy body; PPO (GAE, observation
  normalisation, several worlds side by side), neuro-evolution (the whole
  population evaluated as one batched matrix multiply), DQN (continuous controls
  turned into a menu) and tabular Q-learning share one session layer for charts,
  replays, Skip and "every N" blocks. Small policy networks run single-threaded —
  far faster than letting PyTorch spread a 64-unit layer over 10 cores.
* **Replays, not video.** Runs send compact replays (body poses per frame, rounded,
  thinned to ~1 MB) and the browser draws them, so a whole evolving population can
  be shown as ghosts; the same renderer powers live keyboard play.
* **Connection rules are tested.** Blockly refuses to load stacks whose neighbours
  have incompatible connection checks; `compiler/validate.py` mirrors the rule so
  every example and toolbox block is verified by the test suite.
* **Beginner-proofing.** Shape inference, auto output layers, auto-flatten,
  left-padded text (so "last step" sees real words), friendly error messages
  mapped back to the failing block, quick-test mode, `check` blocks that don't
  count in quick mode, drawing-pad input normalised like the training data.
