# Chapter 11 — Training Deep Neural Networks: Practical Labs

A self-directed lab notebook. Work through these in order — each section maps to a
part of the chapter.

**How each lab is written:**
- **Point** — what concept the lab exists to teach you, so you always know *why* you're
  doing it. (Read this first if a task feels aimless.)
- The task itself is intentionally loose — figure out the *how* yourself.
- **Wonder** — an open question to chase once the lab works. Not required, but it's where
  the real understanding lives.

**Suggested setup:** PyTorch, a small dataset you can iterate on fast (Fashion-MNIST
from Ch. 10 is perfect; CIFAR10 if you want color), matplotlib for plots, and a GPU
if you have one. Keep one reusable training loop and one reusable "build an N-layer
MLP" helper — you'll swap pieces in and out constantly.

---

## Part 1 — Vanishing/Exploding Gradients & Weight Initialization

**Lab 1.1 — See saturation for yourself.**
*Point:* during backprop, the gradient gets multiplied by the activation's derivative
at every layer. Sigmoid's derivative is near zero almost everywhere — so the gradient
dies before reaching the lower layers. This lab makes that cause visible to your eyes.
Plot the sigmoid function and its derivative on the same axes over a *symmetric* input
range (e.g. −10 to 10), and mark where the derivative is effectively zero.
*Wonder:* at what input magnitude does the gradient practically vanish? Repeat for
tanh — why is a mean of 0 supposed to behave "slightly better" than a mean of 0.5?

**Lab 1.2 — Watch variance drift through a deep stack.**
*Point:* bad initialization lets the signal's variance snowball (explode) or collapse
(vanish) as it passes through layers; good init is *designed* to hold variance roughly
constant in both directions. You're watching that happen, layer by layer.
Build a deep stack of linear layers with a saturating activation, initialize the
weights from a standard normal (mean 0, std 1), feed in standardized random input,
and measure the variance of the activations at every layer. Then re-run with Glorot
and He initialization.
*Wonder:* does variance grow, shrink, or hold steady in each case? Can you predict
the trend before you run it?

**Lab 1.3 — Re-derive the initialization table.**
*Point:* the init "rules" aren't magic — each is just a target weight variance derived
from a layer's fan-in and fan-out. Deriving them yourself turns a black box into a
one-line formula.
Using only `fan_in` and `fan_out`, compute the target weight variance for the three
schemes in the chapter (Glorot/Xavier, He/Kaiming, LeCun) by hand. Sample weights
yourself from both the normal and uniform forms, then confirm your numbers match
PyTorch's `torch.nn.init` functions.
*Wonder:* when does LeCun init coincide exactly with Glorot init?

**Lab 1.4 — Fix PyTorch's default and apply it everywhere.**
*Point:* frameworks don't always default to the best init (PyTorch's `nn.Linear`
deliberately doesn't), so being able to override it — for one layer and for a whole
model — is a real, frequently-needed skill.
Correct a single layer's weights two ways: (a) by manually scaling its `weight.data`
and zeroing the bias, and (b) by calling the proper `torch.nn.init` function. Then
write a small function that re-initializes *every* linear layer in a model in one shot.
*Wonder:* why zero the biases instead of randomizing them?

**Lab 1.5 — Norm-preserving init.**
*Point:* an orthogonal matrix preserves the norm of whatever passes through it — a
completely different route to keeping the signal stable, useful where Glorot/He
struggle (e.g. recurrent nets).
Initialize a square layer with an orthogonal matrix and empirically verify that it
preserves the norm of its input vector. Then try a non-square layer.
*Wonder:* what breaks (or doesn't) when the layer isn't square, and where might this
init shine over Glorot/He?

**Lab 1.6 — Calm the output layer.**
*Point:* a network that is wildly overconfident at step 0 produces huge loss gradients
that thrash the weights randomly. Shrinking the output layer's initial weights keeps
the early predictions humble and the gradients sane.
Train a classifier twice: once normally, once after scaling down the output layer's
initial weights. Log the loss and the gradient magnitudes for the first few steps.
*Wonder:* what happens to the spread of the initial predicted probabilities, and why
might "less confident at the start" be a good thing?

---

## Part 2 — Better Activation Functions

**Lab 2.1 — Kill some neurons.**
*Point:* ReLU neurons can *permanently* die — get stuck outputting zero for every
input, with zero gradient to ever recover — and a too-large learning rate is the usual
trigger. You're inducing the failure so you can recognize it in the wild.
Train a ReLU network with a deliberately large learning rate. Define and measure
"dead neurons" (those outputting zero for every instance in a batch). Report the
dead fraction.
*Wonder:* can a dead neuron ever come back? Construct a scenario where one does.

**Lab 2.2 — One plot, every activation.**
*Point:* every activation is a different bundle of trade-offs — saturation, smoothness,
compute cost, whether it's zero-centered. Seeing them overlaid builds the intuition for
*why* you'd reach for one over another.
Plot ReLU, Leaky ReLU, ELU, SELU, GELU, Swish/SiLU, Mish, and ReLU² together over a
shared input range. Reproduce the chapter's observations about where Mish overlaps
Swish and where it overlaps GELU.
*Wonder:* which of these are non-convex and/or non-monotonic, and can you spot it
visually?

**Lab 2.3 — Leaky family, initialized correctly.**
*Point:* leaky variants never fully die (they keep a small slope for negatives), and
because that slope changes the variance, the *init* has to be tuned to match. This lab
ties the activation choice and the init choice together.
Use the leaky-ReLU variants and pair them with the *correctly adjusted* Kaiming init.
Then make the leak slope a learned parameter instead of a fixed one.
*Wonder:* on a small dataset, does the learnable slope help or overfit?

**Lab 2.4 — Make a network self-normalize.**
*Point:* SELU can make a plain MLP keep its activations centered at mean 0 / std 1 all
on its own — solving vanishing/exploding gradients without BN — but only when several
strict conditions all hold. You're proving both the effect *and* how fragile it is.
Build a pure MLP that satisfies all the self-normalization conditions with SELU, and
track the mean and standard deviation of each layer's outputs across training. Then
deliberately violate one condition and watch self-normalization fall apart.
*Wonder:* which single condition, when broken, does the most damage?

**Lab 2.5 — Build the gated and squared ones by hand.**
*Point:* modern activations like SwiGLU aren't single curves — they gate one signal
with another by splitting and multiplying. Building it by hand demystifies what's
actually inside a transformer's feed-forward block.
Implement SwiGLU from scratch (hint: it needs the previous layer to produce twice as
many outputs, which you then split). Separately, implement ReLU² directly.
*Wonder:* how does SwiGLU change the parameter count compared to a plain activation,
and is that a fair comparison?

**Lab 2.6 — Tune the Swish knob.**
*Point:* an activation's *shape* can itself be a learnable parameter, not a fixed
choice. You're letting gradient descent pick the shape and seeing where it lands.
Compare fixed-β Swish against a version where β is learned. Check whether a learned
β drifts toward the value that makes Swish approximate GELU.
*Wonder:* one β for the whole model vs one per layer — what's the trade-off?

**Lab 2.7 — Fast approximations.**
*Point:* the "hard" activation variants trade a little accuracy for real speed — which
is exactly the trade you make when shipping to phones or edge devices.
Benchmark a "hard" approximation activation against its smooth counterpart for both
forward-pass speed and output difference.
*Wonder:* where would the speed matter enough to accept the approximation?

**Lab 2.8 — Activation bake-off.**
*Point:* folklore makes a big deal of activation choice, but in practice it often
matters less than expected. You're testing that claim instead of taking it on faith.
Hold architecture, data, and optimizer fixed; swap only the activation; compare
convergence curves.
*Wonder:* does the "best" activation depend on network depth?

---

## Part 3 — Batch Normalization

**Lab 3.1 — Let BN do the scaling.**
*Point:* a batch-norm layer placed first can absorb the job of input standardization
entirely. You're testing whether "just let BN do it" really substitutes for a manual
StandardScaler.
Build an image classifier with a batch-norm layer as the very first layer (after
flattening) and skip manual input standardization. Compare against a version that
standardizes inputs the old way.
*Wonder:* how close is "approximately standardized by BN" to the real thing?

**Lab 3.2 — Look inside a BN layer.**
*Point:* a BN layer secretly holds two *different kinds* of state — learned parameters
(scale/shift) updated by backprop, and tracked running statistics updated by a moving
average. Seeing them separately stops BN from being a mystery box.
Inspect a trained BN layer's learnable parameters and its buffers separately. Identify
which correspond to scale/shift and which to the running statistics, and watch the
running stats evolve over training.
*Wonder:* which of these update via backprop and which via a moving average?

**Lab 3.3 — Before or after?**
*Point:* there's a genuine, unsettled debate about whether BN belongs before or after
the activation — and the choice even decides whether the preceding layer's bias is
redundant. You're forming your own opinion from data, not dogma.
Build the same network with BN before the activation and after the activation. When
you put BN before, also drop the bias from the preceding linear layer.
*Wonder:* why is the linear layer's bias redundant in that arrangement?

**Lab 3.4 — The most common mistake.**
*Point:* BN computes differently in training vs evaluation mode, and forgetting to
switch is one of the most common real-world PyTorch bugs. You're feeling the
consequence on purpose so it never bites you silently.
Train a BN model, then make predictions while *forgetting* to switch to evaluation
mode. Compare against doing it correctly.
*Wonder:* why do single-instance predictions suffer the most?

**Lab 3.5 — The momentum knob.**
*Point:* BN's momentum controls how fast its running statistics chase the data, and the
right value depends on your batch size. You're seeing that dependence directly.
Vary the BN momentum hyperparameter and observe how quickly the running statistics
track the data, for both a large and a small batch size.
*Wonder:* PyTorch's BN "momentum" means the opposite of the usual convention —
which weight does it actually control?

**Lab 3.6 — 1D, 2D, and sequences.**
*Point:* "batch norm" pools its statistics over different dimensions for vectors,
images, and sequences. You're learning which axes get normalized in each case (and why
sequences need a rearrange first).
Apply 2D batch norm directly to image batches (before flattening), and apply 1D batch
norm to a batch of sequences (you'll need to rearrange the dimensions first).
*Wonder:* across which dimensions are the statistics pooled in each case, and how many
scale/shift parameters end up per channel?

**Lab 3.7 — Fold BN away.**
*Point:* BN adds a runtime cost, but once trained you can often fuse it into the
previous linear layer for free — a standard inference optimization. You're performing
the fusion and verifying it changes nothing but speed.
After training a network with BN before the activation, fuse each BN layer into its
preceding linear layer by computing new weights and biases. Verify the fused model
produces (nearly) identical outputs, then benchmark inference speed.
*Wonder:* why can you only fold it cleanly when BN sits right after a linear layer?

**Lab 3.8 — Does it actually help?**
*Point:* BN's headline selling point is faster *convergence*, even though each epoch
gets slower. You're checking whether the net effect (fewer epochs × slower epochs) is
actually a win on wall-clock time.
Compare a deep network with and without BN: epochs to a target accuracy, and total
wall-clock time.
*Wonder:* per-epoch time goes up with BN — so why can total time go down?

---

## Part 4 — Layer Normalization

**Lab 4.1 — Reproduce LN by hand.**
*Point:* layer norm is "just" normalizing over the feature dimension for each instance
independently — no batch involved. Reproducing it manually proves there's nothing
magical happening.
Run inputs through PyTorch's layer-norm module, then reproduce the exact same result
with manual mean/variance computation over the normalized dimensions.
*Wonder:* which dimensions are you averaging over, and why is that "per instance"?

**Lab 4.2 — LN vs BN under pressure.**
*Point:* LN sidesteps BN's two biggest weaknesses — it doesn't care about batch size,
and it behaves identically in train and eval mode. You're demonstrating both, which is
exactly why transformers prefer it.
Compare layer norm and batch norm at very small batch sizes, and check that LN behaves
identically in training and evaluation mode while BN does not.
*Wonder:* why does LN not need running statistics at all?

**Lab 4.3 — What to normalize over.**
*Point:* "the features" you normalize over is a design decision, not a given. You're
seeing how the choice plays out for images (per-channel vs across all channels).
For an image batch, normalize each channel independently vs normalizing across all
channels at once, and compare.
*Wonder:* which choice do most vision architectures prefer, and why might that be?

---

## Part 5 — Gradient Clipping

**Lab 5.1 — Two ways to clip.**
*Point:* clipping by value and clipping by norm do different things to the gradient's
*direction*, not just its size — one can reorient the update, the other can't. You're
seeing the geometric difference.
Take a gradient vector that points strongly along one axis (e.g. one tiny component,
one huge component) and clip it by value, then by norm. Compare the resulting vectors.
*Wonder:* which method preserves direction, and which reorients the vector toward the
diagonal?

**Lab 5.2 — Tame an explosion.**
*Point:* clipping is the standard cure for exploding gradients (especially in recurrent
nets). You're deliberately causing the disease, then applying the cure.
Induce exploding gradients (very deep net and/or large learning rate), then add
clipping into the training loop right after backprop and observe the difference.
*Wonder:* does the best clip type/threshold depend on the data?

---

## Part 6 — Reusing Pretrained Layers (Transfer Learning)

**Lab 6.1 — Transfer the chapter's way.**
*Point:* reusing a pretrained network's lower layers can crush from-scratch training
when labeled data is tiny — but the benefit is fragile and dangerously easy to oversell
(the author admits to cherry-picking a seed). You're both *using* transfer learning and
*stress-testing the hype*.
Train a model on a subset of classes ("task A"). Then tackle a tiny related task
("task B", binary, very few labeled examples): first from scratch, then by deep-copying
and reusing A's lower layers under a new output head. Freeze the reused layers, train
the new head, then unfreeze and fine-tune at a lower learning rate.
*Wonder:* re-run with different seeds and class pairs — how often does the improvement
survive? Why does transfer learning work poorly for small dense networks specifically?

**Lab 6.2 — Different speeds for different layers.**
*Point:* different parts of a model can and often *should* learn at different rates —
this is the everyday mechanic of fine-tuning. You're wiring up per-group learning rates.
Set up a single optimizer that updates the freshly added layers faster than the reused
ones, using parameter groups.
*Wonder:* why is "new layers fast, reused layers slow" a sensible default in transfer
learning?

---

## Part 7 — Unsupervised & Auxiliary-Task Pretraining

**Lab 7.1 — Pretrain without labels.**
*Point:* when labels are scarce but raw data is plentiful, you can learn useful features
*first* with no labels at all (e.g. via an autoencoder), then fine-tune on the few
labels you have. You're measuring how much that head start is worth.
Train an autoencoder on unlabeled data, then reuse its lower (encoder) layers, add a
task head, and fine-tune on a *small* labeled set. Compare against training the same
classifier from scratch on just the small labeled set.
*Wonder:* how small does the labeled set have to get before pretraining clearly wins?

**Lab 7.2 — Invent your own labels (self-supervised).**
*Point:* you can manufacture labels straight out of the data (hide part of it, predict
the hidden part) — this is the core idea behind how modern LLMs are pretrained. You're
building a tiny version of that.
Take an unlabeled dataset and automatically generate a pretext task (e.g. mask part of
each input and predict the masked part). Pretrain on it, then transfer to a real
downstream task.
*Wonder:* what makes a *good* pretext task — what should the model be forced to learn?

---

## Part 8 — Faster Optimizers

**Lab 8.1 — Momentum and Nesterov, from scratch.**
*Point:* momentum and Nesterov speed up descent by *remembering* past gradients, like a
ball gathering speed downhill. Coding them by hand on a hard surface shows concretely
why they beat plain gradient descent.
On a badly-scaled quadratic "elongated bowl", implement plain gradient descent,
momentum, and Nesterov accelerated gradient yourself, and plot all three trajectories
toward the minimum.
*Wonder:* verify the terminal-velocity claim — does the step size really settle at the
gradient times the learning rate times 1/(1−β)?

**Lab 8.2 — AdaGrad's strength and flaw.**
*Point:* AdaGrad adapts the step size per-dimension, which lets it aim at the optimum
earlier — but because it accumulates *forever*, it eventually starves its own learning
rate and stops short. You're seeing both halves of that story.
Implement AdaGrad on the same bowl and watch it correct its direction early — then keep
training and watch it stall before reaching the optimum.
*Wonder:* what exactly causes it to grind to a halt?

**Lab 8.3 — RMSProp to the rescue.**
*Point:* RMSProp is AdaGrad with *forgetting* — one small change (a decaying average
instead of a running sum) removes the stall entirely. You're isolating that single
change and its outsized effect.
Modify your AdaGrad code into RMSProp (swap the running sum for an exponentially
decaying average). Show it no longer stalls.
*Wonder:* what role does the decay rate play, and what happens at the extremes?

**Lab 8.4 — Adam, by hand.**
*Point:* Adam is essentially momentum + RMSProp + a startup bias correction. Building it
yourself shows it's a *combination* of ideas you already understand, not a new mystery.
Implement Adam from scratch including the bias-correction steps, and compare your
trajectory and final result against PyTorch's built-in Adam.
*Wonder:* what would go wrong early in training if you dropped the bias correction?

**Lab 8.5 — Reproduce the comparison table.**
*Point:* optimizers trade convergence *speed* against final *quality* differently, and
the chapter's star table is the summary. You're regenerating that table from your own
runs instead of trusting it.
Run the full roster (plain SGD, SGD+momentum, SGD+Nesterov, AdaGrad, RMSProp, Adam,
AdaMax, NAdam, AdamW) on one task and rate each on convergence speed and final quality.
*Wonder:* which optimizers landed differently from the table, and can you guess why?

**Lab 8.6 — Adaptive isn't always better.**
*Point:* two uncomfortable truths — adaptive optimizers can *generalize worse* than
plain SGD on some datasets, and ℓ2 regularization stops being equivalent to weight decay
the moment you leave SGD. You're confronting both directly.
Compare AdamW against Adam-with-classic-ℓ2, and compare both against Nesterov on a
dataset where they're "allergic to adaptive gradients" (watch generalization, not just
training loss).
*Wonder:* why are ℓ2 regularization and weight decay equivalent under SGD but not under
Adam?

**Lab 8.7 — A second-order approximation.**
*Point:* true second-order methods (using the Hessian/curvature) would be wonderful but
are far too expensive for big nets; Shampoo cheaply *approximates* that information.
You're trying the approximation and learning why the real thing is off the table.
Install and try the Shampoo optimizer (it lives outside core PyTorch) on a small model.
*Wonder:* why are true Hessian-based methods impractical for DNNs, and what is Shampoo
approximating instead?

---

## Part 9 — Training Sparse Models

**Lab 9.1 — Prune after training.**
*Point:* you can shrink a finished model by zeroing its smallest weights, or by removing
whole neurons/channels — trading a little accuracy for speed and memory. You're
measuring that trade-off curve.
Train a normal (dense) model, then zero out the smallest weights (unstructured pruning)
and, separately, remove whole neurons/channels (structured pruning). Measure the
resulting sparsity and the accuracy cost.
*Wonder:* at what sparsity level does accuracy fall off a cliff?

**Lab 9.2 — Encourage sparsity during training.**
*Point:* ℓ1 regularization actively pushes weights to *exactly* zero while you train, so
you get a sparse model for free rather than pruning afterward. You're confirming it
really does zero things out.
Add ℓ1 regularization to push weights toward zero as you train, then count how many
weights ended up effectively zero compared to an unregularized run.
*Wonder:* how does the ℓ1 strength trade off sparsity against accuracy?

---

## Part 10 — Learning Rate Scheduling

**Lab 10.1 — Plot the schedules first (no training).**
*Point:* before judging schedules by their results, you need a mental picture of their
*shapes*. Stepping each one with no training and plotting the rate vs epoch builds that
vocabulary cheaply.
Step each scheduler in a loop and plot the learning rate vs epoch: exponential, cosine
annealing, cosine annealing with warm restarts, 1cycle, and linear warm-up. Reproduce
the shapes from the chapter's figures.
*Wonder:* which schedules keep the rate high longer, and when would you want that?

**Lab 10.2 — Exponential decay.**
*Point:* exponential decay is the simplest schedule and its behavior is fully
predictable from one number. You're verifying the prediction against reality.
Use exponential scheduling and verify the textbook claim numerically (with a decay
factor of 0.9, the rate should be roughly a third of the original after 10 epochs).
*Wonder:* how do you pick the decay factor for a known number of epochs?

**Lab 10.3 — Cosine annealing.**
*Point:* cosine annealing holds the rate high for most of training, then eases it down
near the end — often outperforming exponential. You're trying it and feeling the
difference.
Train with cosine annealing from a max rate down to a small minimum.
*Wonder:* the chapter calls choosing the schedule's length awkward — why?

**Lab 10.4 — React to a plateau.**
*Point:* instead of committing to a fixed schedule in advance, you can *react* — drop
the rate only when a validation metric stalls. You're building that feedback loop.
Use performance/adaptive scheduling that drops the rate when a validation metric stops
improving. Wire the metric into the scheduler each epoch.
*Wonder:* how do the patience and factor settings change the training curve?

**Lab 10.5 — Warm up, then hand off.**
*Point:* the first few steps of training can be chaotic; warming the rate up from near
zero stabilizes them before a decay schedule takes over. You're chaining two schedulers
and seeing why the warm-up phase exists.
Linearly warm the learning rate up over the first few epochs, then deactivate the
warm-up and let your main scheduler take over. Reproduce the same warm-up two ways (a
built-in linear schedule and a custom lambda).
*Wonder:* why does warm-up help "sensitive" models or very large batch sizes?

**Lab 10.6 — Warm restarts.**
*Point:* periodically shooting the learning rate back up lets training jump out of
plateaus and local optima on its own. You're watching that escape mechanism work.
Use cosine annealing with warm restarts, doubling the cycle length each round.
*Wonder:* how do the periodic rate spikes help escape plateaus?

**Lab 10.7 — Chase super-convergence.**
*Point:* a well-shaped schedule (1cycle) can sometimes reach a target accuracy in *far*
fewer epochs — the effect nicknamed "super-convergence." You're testing whether the
claim holds on your setup.
Train with 1cycle scheduling and compare epochs-to-target-accuracy against a plain
constant rate on the same architecture.
*Wonder:* 1cycle cools momentum as it heats the learning rate (and vice versa) — why
coordinate them inversely?

---

## Part 11 — Regularization

**Lab 11.1 — ℓ2 three ways.**
*Point:* ℓ2 / weight decay can be applied globally, selectively, or per parameter-group
— and *which* parameters you decay (you usually want to spare biases and norm-layer
params) genuinely matters. You're learning all three mechanics.
Apply ℓ2 regularization (a) via the optimizer's weight-decay argument, (b) by computing
the ℓ2 penalty manually in the loss for selected parameters only, and (c) via parameter
groups so that biases and norm-layer parameters are excluded.
*Wonder:* why might you *not* want to decay biases and BN/LN parameters?

**Lab 11.2 — ℓ1 by hand.**
*Point:* PyTorch gives you no built-in ℓ1 helper, and ℓ1 produces a *different* kind of
regularization than ℓ2 — it drives weights to exactly zero. You're implementing it and
feeling that difference.
Add an ℓ1 penalty to the loss yourself.
*Wonder:* why does ℓ1 push weights to exactly zero while ℓ2 only shrinks them?

**Lab 11.3 — Dropout dial.**
*Point:* dropout fights overfitting by forcing neurons not to depend on each other, and
the dropout rate is the main knob. You're learning to *read* over- vs under-fitting off
the training/validation curves.
Add dropout before each layer, train it (minding train/eval mode), and evaluate the
*training* loss with dropout off so the comparison to validation loss is honest. Sweep
the dropout rate from low to high.
*Wonder:* what do over- and under-fitting look like as you turn the dial, and should big
and small layers get the same rate?

**Lab 11.4 — Monte Carlo dropout.**
*Point:* leaving dropout *on* at prediction time turns a single trained model into an
ensemble — averaging many noisy passes gives better predictions *and* a free uncertainty
estimate. You're building MC dropout and seeing what uncertainty actually looks like.
Take a trained dropout model, keep dropout active at prediction time, run many
stochastic forward passes on the same inputs, and average the predicted probabilities.
Also compute the per-class standard deviation as an uncertainty estimate.
*Wonder:* the chapter warns against averaging logits and *then* applying softmax — try
it the wrong way and see how overconfident it gets. How many MC samples before extra
ones stop helping?

**Lab 11.5 — A cleaner MC dropout.**
*Point:* toggling train/eval to force dropout on at inference is brittle and breaks other
machinery; a tiny always-on dropout module is the clean fix. You're replacing the hack.
Replace the train/eval hack with a tiny custom dropout module that is always active.
*Wonder:* why is the hack "brittle" — what does it break?

**Lab 11.6 — Dropout for self-normalizing nets.**
*Point:* ordinary dropout *destroys* SELU's self-normalization, so there's a special
variance-preserving variant for that case. You're learning to match the regularizer to
the activation.
On a SELU self-normalizing network, use the variance-preserving dropout variant instead
of regular dropout.
*Wonder:* why would ordinary dropout destroy self-normalization?

**Lab 11.7 — Max-norm.**
*Point:* max-norm regularizes by directly *capping* each neuron's incoming weight
magnitude — a hard constraint applied after each step, rather than a penalty added to
the loss. You're implementing that constraint.
After each optimizer step, rescale any neuron's incoming weights whose ℓ2 norm exceeds a
threshold. Sweep the threshold and observe the regularization strength.
*Wonder:* why is max-norm applied *after* the step rather than added to the loss, and how
does the rescaling dimension change for convolutional layers?

---

## Part 12 — Capstone: Practical Guidelines

**Lab 12.1 — Assemble the defaults.**
*Point:* the chapter's recommended defaults are a recipe that works well with little
tuning — but the real skill is *adapting* the recipe to a goal. You're assembling the
recipe, then specializing it three ways.
Build one training pipeline that combines the chapter's recommended defaults (good init,
a sensible activation by depth, normalization, weight decay + early stopping, a strong
optimizer, and a learning-rate schedule). Run it on a real dataset with minimal tuning.
*Wonder:* now produce three variants tuned for (a) a sparse model, (b) a low-latency
model, and (c) a risk-sensitive model that needs trustworthy uncertainty. What do you
change in each, and what do you sacrifice?