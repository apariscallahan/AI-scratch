# NeuroBlocks block reference

## Start

| block | what it does |
|---|---|
| `when [flag] clicked` | Like Scratch's green flag: everything stacked under this block runs, top to bottom, when you press Run. You can have several of these; they run one after another from the top of the workspace down. |
| `use device [device] random seed [seed]` | 'auto' picks an NVIDIA GPU if there is one, then an Apple GPU, then the CPU. A seed makes random things (initial weights, shuffling, terrain) the same every run. On the command line, --device overrides this block. |
| `every [n] [unit] do [do]` | Goes inside a 'train' block. The blocks inside run every N steps (or epochs), so you can watch the model improve as it learns. |

## Data

| block | what it does |
|---|---|
| `make toy dataset [name] : [kind] [settings]` | Toy datasets are made of dots on a plane. Classification ones (spirals, moons, circles…) have coloured dots to separate; the '→ number' ones are curves to fit. Use 'show decision map' to watch what a model has learned. |
| `load table [name] : [kind] [settings]` | Each row is one example (a flower, a wine, a patient…) described by a few numbers. The model learns to predict the last column: a class or a number. |
| `load images [name] : [kind] [settings]` | Digits 8×8 is tiny and built in. MNIST (handwritten digits) and Fashion-MNIST (clothes) are 70,000 28×28 grey pictures. CIFAR-10 is 60,000 colour photos of planes, cars, animals… Downloads are cached. |
| `load text [name] : [kind] [settings]` | Language models read text one token at a time and learn to guess the next one. After training they can write new text in the same style. 'context length' is how many tokens they can look back at. |
| `load labelled texts [name] : [kind] [settings]` | Short texts with a label each (e.g. positive/negative reviews) for text classification. |
| `make time series [name] : [kind] [settings]` | A signal over time, cut into windows: look at the past, predict what comes next. |
| `load CSV file [file] as [name] predict column [target] [settings]` | Upload a CSV from the Data tab (it goes into the 'data' folder). Words in a column are turned into numbers automatically, missing values are filled in, and the task (classes or numbers) is guessed from the column you predict. |
| `load text file [file] as [name] [settings]` | Train a language model on your own .txt file (stories, song lyrics you wrote, code …). |
| `load text from web address [url] as [name] [settings]` | Download a plain-text page (e.g. a public-domain book) and use it as a language-model dataset. |
| `load image folder [folder] as [name] [settings]` | Your own pictures: one sub-folder per class (my_pictures/cats, my_pictures/dogs …). |
| `load Hugging Face dataset [id] as [name] text column [text] label column [label] [settings]` | Any dataset from huggingface.co (needs: pip install datasets). With a label column it becomes text classification; without, a language-model text. |
| `keep [pct] % aside for testing` | Hold back some examples the model never trains on, to test it fairly. |
| `use [n] examples` | How many examples to make (toy data) or to use (others — fewer is faster). |
| `noise [noise]` | How messy the toy data is (0 = perfectly clean). |
| `number of classes [n]` | How many groups/colours (spirals and blobs). |
| `standardize numbers [on]` | Rescale every column to average 0 and spread 1 — helps most models learn. |
| `split text into [tok]` | Tokens are the pieces a language model reads: single characters, whole words, or GPT-2's sub-word pieces. |
| `context length [n] tokens` | How many tokens the model sees at once. Longer = more memory and slower, but better long-range understanding. |
| `keep the [n] most common words` | Vocabulary size for word tokens; rarer words become <unk>. |
| `make text lowercase [on]` | Treat 'The' and 'the' as the same. |
| `resize images to [n] pixels` | Make every image N×N pixels (smaller = faster). |
| `make images grey [on]` | Drop colour (3 channels → 1). |
| `augment images with [aug]` | Randomly change training images a little each time, so the model can't just memorise them. |
| `look at [w] past values, predict [h] next` | Window sizes for time series. |
| `task [task]` | For CSV files: is the column you predict a category or a number? |
| `shuffle with seed [seed]` | Change how examples are shuffled and split (same seed = same split). |
| `plot [data] as dots` | Scatter plot of the dataset (big tables are squashed to 2-D with PCA). |
| `show [n] examples from [data]` | Show some examples (pictures, rows or text) in the Results tab. |
| `draw a map of [data] using [method]` | Squash many numbers per example into a 2-D map where similar examples land close together. |
| `number of examples in [data]` | How many examples the dataset has (training + test). |
| `number of classes in [data]` | How many different answers (classes) the dataset has. |

## Neural Nets

| block | what it does |
|---|---|
| `create neural network [name] [layers]` | Data flows from the top layer to the bottom one. You never have to say how big the input is — it's worked out from the data. End with an 'output layer'; it sizes itself to the number of classes (or numbers, or tokens) to predict. |
| `create GPT language model [name] layers [layers] heads [heads] embedding size [embed] dropout [dropout]` | layers = how many transformer blocks are stacked (depth). heads = how many things each layer can pay attention to at once. embedding size = how many numbers describe each token (width; must divide by heads). More of each = smarter but slower. A cloud GPU can handle 8–12 layers with embedding 384–768. |
| `load pretrained language model [modelid] as [name]` | Download a pretrained model from Hugging Face (e.g. distilgpt2, gpt2) to fine-tune on your text. Needs: pip install transformers. Best on a GPU. |
| `create picture generator [name] using [kind]` | A VAE squeezes pictures into a few numbers and learns to turn random numbers back into pictures. A GAN has two players: an artist that draws and a critic that tries to spot fakes; they both get better. Both learn the classes too, so you can ask for 'a 7'. |
| `show [model]` | Show the model's layers, their sizes and parameter counts in the Model tab. |
| `reset [model]` | Forget everything the model learned (it starts again from random weights). |
| `number of parameters in [model]` | How many numbers (weights) the model learns. |

## Layers

| block | what it does |
|---|---|
| `dense layer [units] units [act]` | A dense (fully connected) layer has 'units' neurons. Each neuron adds up all its inputs times learned weights, then applies the activation (ReLU keeps positives, zeroes negatives) so the network can learn curvy shapes. |
| `output layer (sized automatically)` | The last layer: one unit per class / number / token. Its size is set from the data. |
| `activation [act]` | Apply a non-linear function to every number. |
| `dropout [rate]` | While training, randomly switch off this fraction of signals so the network doesn't over-rely on any one of them (reduces overfitting). |
| `batch normalization` | Re-centre and re-scale signals using statistics of each batch — often trains faster. |
| `layer normalization` | Re-centre and re-scale each example's signals (used in transformers). |
| `flatten` | Turn an image (channels × height × width) into one long list of numbers. |
| `conv 2D [filters] filters size [size] stride [stride] [act]` | Each filter is a small window (e.g. 3×3) of learned weights that scans across the picture. Early conv layers find edges; deeper ones find eyes, wheels, letters… 'filters' = how many different patterns to look for. Stride 2 halves the image size. |
| `[kind] pool [size]` | Shrink the image by keeping the biggest (or average) value of each little square. |
| `global [kind] pool` | Squash each channel of an image (or each feature of a sequence) to a single number. |
| `upsample ×[scale]` | Make the image bigger (each pixel becomes a square) — used in decoders / generators. |
| `reshape to [shape]` | Rearrange numbers into a new shape, e.g. 16,7,7 = 16 channels of 7×7 pictures. |
| `token embedding [dim] numbers per token` | Turn each token (character/word id) into a list of learned numbers. Start text models with this. |
| `positional embedding` | Tell the model where each token is in the sequence (transformers need this). |
| `transformer block [heads] heads dropout [drop] [mask]` | The building block of GPT, BERT and friends. 'heads' are parallel attention patterns. For language models (auto) a token may only look at earlier tokens — it can't peek at the answer. Put several in a 'repeat' block to make the model deeper. |
| `[kind] layer [units] units [keep]` | A recurrent layer reads a sequence step by step, carrying a memory along. |
| `combine steps by [mode]` | Turn a sequence of vectors into one vector (e.g. before the output layer of a text classifier). |
| `repeat [times] times [body]` | Stack copies of the layers inside (each copy learns its own weights). Great for deep networks. |
| `residual shortcut around [body]` | Add the input back to the output of the layers inside (a 'skip connection', as in ResNet). Helps very deep networks train. |

## Training

| block | what it does |
|---|---|
| `train [model] on [data] [settings]` | Training shows the model batches of examples, measures how wrong it is (the loss) and nudges every weight to be a little less wrong. One epoch = one pass over all training examples. Watch the Charts tab: training loss should go down; if test loss goes up while training loss goes down, the model is memorising (overfitting). |
| `for [n] epochs` | How many times to go through all the training examples. |
| `for [n] steps` | Train for an exact number of steps (batches). Language models use steps. |
| `batch size [n]` | How many examples the model looks at before each update. |
| `optimizer [opt] learning rate [lr]` | The method that updates the weights, and how big each step is. |
| `learning rate [lr]` | How big each learning step is (too big = chaos, too small = slow). |
| `learning rate schedule [sched] warm up [warm] steps` | Change the learning rate during training: start gently (warm-up), then slowly lower it. |
| `weight decay [wd]` | Gently pull weights towards zero to reduce overfitting. |
| `loss [loss]` | How mistakes are measured. 'auto' picks the right one for the task. |
| `clip gradients at [clip]` | Limit how big a single update can be (keeps training stable). |
| `stop early after [n] checks without improvement` | Stop training when the test loss stops getting better. |
| `check progress every [n] steps` | How often a language model is tested on validation text. |
| `goal [goal]` | What to learn. 'rebuild the input' makes an autoencoder. |
| `mixed precision [mode]` | On GPUs, compute with 16-bit numbers: faster, less memory. |
| `compile the model for speed [on]` | Use torch.compile on NVIDIA GPUs (slow start, faster steps). Ignored on CPU. |
| `name this run [label]` | The name used for this run's lines in the charts. |
| `save the best version to file [file]` | Whenever the test score improves, save the model. |
| `[which] of [model]` | A number from the model's training history (use it in 'say', charts or 'check that'). |

## Test & Play

| block | what it does |
|---|---|
| `test [model] on [data]` | Score the model on the test examples it never trained on: accuracy, errors, confusion matrix. |
| `[metric] of [model] on [data]` | A test score as a number (accuracy is between 0 and 1). |
| `check that [value] [op] [target]` | A test: passes (✅) or fails (❌). On the command line, failed checks make the run exit with an error — useful for automatic testing. |
| `show predictions of [model] on [n] examples from [data]` | See what the model guesses for some test examples (green = right, red = wrong). |
| `show confusion matrix of [model] on [data]` | A grid showing which classes get mixed up with which. |
| `show decision map of [model] on [data]` | Colour the whole plane by the model's prediction (for 2-D toy data). Put it in 'every N steps' to watch the boundary form. |
| `prediction of [model] for [input]` | Ask the model about one example: a list of numbers, or a text. |
| `write [length] tokens with [model] starting with [prompt] creativity [temp]` | Let a language model write. Creativity (temperature): 0 = always the most likely token, 1 = adventurous, 2 = chaotic. |
| `text written by [model] starting with [prompt] length [length]` | The text a language model writes (as a value you can 'say' or store). |
| `show [n] new pictures from [model] of class [class]` | Ask a picture generator to invent new pictures (class 'any', or a class name such as 7 or sneaker). |
| `let me prompt [model]` | Opens a prompt box in the Play tab so you can type and see what the model writes. |
| `let me draw for [model]` | Opens a drawing pad in the Play tab: draw a digit (or anything) and see the model's guesses live. |
| `let me type inputs for [model]` | Opens a form in the Play tab: type values for each input and get a prediction. |

## Simulations

| block | what it does |
|---|---|
| `create car world [name] [parts]` | Build the car's world from parts (World Parts category): the terrain, what the car CAN DO (abilities become the model's outputs), what it CAN SENSE (senses become the model's inputs), and what counts as doing well (rewards). Reinforcement learning then tries things, keeps what earns reward, and slowly gets better. |
| `create balancing-pole world [name] [parts]` | Classic CartPole: push a cart left or right to keep a pole standing up. |
| `create mountain car world [name] [parts]` | A weak car must rock back and forth to build up speed and reach the flag on the hill. |
| `create pendulum world [name] [parts]` | Swing a pendulum up and balance it upside-down using a weak motor. |
| `create rocket lander world [name] [parts]` | Fire the main and side thrusters to land a rocket gently on the pad. |
| `create maze world [name] [parts]` | A grid maze: find the way from the start to the goal (avoid traps). |
| `create flappy bird world [name] [parts]` | Flap to fly through the gaps between pipes. |
| `train [model] to play in [world] [settings]` | The model's inputs are the world's senses and its outputs are the abilities. PPO is a good all-rounder. Evolution runs a whole population at once and keeps the best — fun to watch. DQN picks between a few actions. Q-table is for small mazes. Replays appear in the Sim tab. |
| `learn with [algo]` | The learning method. |
| `for [n] steps` | How many moments of practice in total (more = better, slower). |
| `for [n] generations` | Evolution: how many rounds of survival of the fittest. |
| `population [n]` | Evolution: how many brains try at the same time. |
| `mutation strength [s]` | Evolution: how much children differ from their parents. |
| `learning rate [lr]` | How big each learning step is. |
| `care about the future [g]` | Discount factor: 0.99 = plan ahead, 0.9 = short-sighted. |
| `run [n] worlds at once` | More copies of the world = more experience per update. |
| `exploration [e]` | How much to try random things (0 = never, 1 = always). |
| `show a replay every [n] updates` | How often to send an animation to the Sim tab. |
| `name this run [label]` | The name used for this run's lines in the charts. |
| `watch [model] play in [world] [n] times` | Replay the trained model in the world (Sim tab). |
| `show [world]` | Preview the world in the Sim tab before training. |
| `let me play [world]` | Control the world yourself with the arrow keys (Play tab). Can you beat the AI? |
| `average score of [model] in [world] over [n] tries` | Total reward per try, averaged (no randomness in the model's choices). |

## World Parts

| block | what it does |
|---|---|
| `terrain [kind] bumpiness [bump] hills [hills] length [len] m` | The ground the car drives on. |
| `car body length [len] wheel size [wheel] weight [w]` | The car's shape and weight. |
| `can drive with [wheels] wheels power [power]` | An ability: spin the wheels forward or back. |
| `can brake` | An ability: lock the wheels. |
| `can lean forwards / backwards strength [s]` | An ability: tilt the car's body (like a rider shifting their weight). |
| `can jump strength [s]` | An ability: hop into the air (needs wheels on the ground). |
| `can rocket boost power [p] fuel [f] seconds` | An ability: a rocket that pushes the car forward. |
| `can sense [what]` | A sense: becomes one of the model's inputs. |
| `can sense the ground ahead with [rays] rays reaching [reach] m` | Like a little radar: measures the distance to the ground in several directions ahead. |
| `reward [what] × [w]` | Points for good behaviour (each moment). |
| `reward reaching the finish +[w]` | Bonus points for reaching the flag. |
| `if it flips over: lose [w] points [end]` | Penalty (and optionally end the try) when the car's body hits the ground. |
| `lose points for [what] × [w]` | Small penalties to encourage efficiency. |
| `end the try if stuck for [secs] seconds` | Stop wasting time when the car isn't moving. |
| `end each try after [secs] seconds` | Time limit for one try (episode). |
| `same world every try [on]` | On: every try uses the same layout. Off: a new random layout each time (harder, but the model learns to handle anything). |
| `gravity [g]` | How strongly things fall (Earth = 9.8, Moon = 1.6). |
| `reward each moment [expr]` | Your own reward formula, built from the world's measurements (the 'world's …' reporter). |
| `end the try when [cond]` | Your own rule for ending a try early. |
| `world's [what]` | A measurement of the world right now — use it inside 'reward each moment' or 'end the try when'. |
| `pole length [l] m` | Longer poles are easier to balance (slower to fall). |
| `wind strength [w]` | Random gusts push the rocket sideways. |
| `fuel [f]` | How much the rocket can fire its engines. |
| `maze size [w] × [h]` | Width and height in squares. |
| `random walls [pct] %` | How much of the maze is wall (always solvable). |
| `[n] traps` | Traps end the try with a penalty. |
| `maze layout [text]` | Draw your own maze: # wall, . floor, S start, G goal, T trap; separate rows with / |
| `gap size [g]` | How big the opening between pipes is. |
| `pipe speed [s]` | How fast the pipes come at you. |

## Classic ML

| block | what it does |
|---|---|
| `create decision tree [name] max depth [depth]` | A flowchart of yes/no questions learned from data. Depth = how many questions in a row (0 = no limit). |
| `create random forest [name] with [trees] trees max depth [depth]` | Many decision trees vote together. Strong and hard to break. |
| `create gradient boosting [name] with [trees] trees learning rate [lr]` | Trees added one at a time, each fixing the previous ones' mistakes. Often the best on tables. |
| `create k-nearest neighbours [name] k = [k]` | Predict by looking at the k most similar training examples. No real 'training' needed. |
| `create linear model [name]` | Draws the best straight line (or flat surface). For classes this is logistic regression. |
| `create logistic regression [name]` | A linear model that outputs class probabilities. |
| `create support vector machine [name] kernel [kernel]` | Finds the widest possible gap between classes. |
| `create naive Bayes [name]` | Simple probability-based classifier. Very fast. |
| `create k-means clustering [name] with [k] groups` | Finds groups in data without being told the answers. |
| `show which inputs matter to [model]` | Bar chart of feature importance (trees, forests, boosting, linear models). |
| `show the questions in decision tree [model]` | Print the learned yes/no questions of a decision tree. |

## Show & Say

| block | what it does |
|---|---|
| `say [text]` | Show a message in the Console (like a Scratch speech bubble). |
| `add point x [x] y [y] to chart [chart] line [series]` | Draw your own live chart: each block adds one point to a line on the Charts tab. |
| `show [kind] chart of [list] titled [title]` | Plot a list of numbers as a chart (Results tab). |
| `play sound [sound]` | Play a sound in the editor — handy at the end of a long training run. |

## Control

| block | what it does |
|---|---|
| `wait [secs] seconds` | Pause the program for a number of seconds. |
| `stop the program` | Stop running right here. |
| `seconds since start` | How many seconds the program has been running. |

## Operators

| block | what it does |
|---|---|
| `round [num] to [digits] decimals` | Round a number to a number of decimal places. |

## Variables

| block | what it does |
|---|---|
| `hyperparameter [var] = [value]` | Works like 'set variable', but when you run the exported program on a cloud machine you can override it: neuroblocks run project.nblk --set steps=50000. Great for scaling a small local experiment up on a GPU. |

## Lists

| block | what it does |
|---|---|
| `add [item] to [var]` | Add an item to the end of a list (starts a new list if the variable is empty). |

## Files

| block | what it does |
|---|---|
| `save [model] to file [file]` | Save a trained model (into the 'models' folder) so you can load it later — e.g. train on a cloud GPU, then test it on your laptop. |
| `load model [name] from file [file]` | Load a model saved earlier (from the 'models' folder). |

