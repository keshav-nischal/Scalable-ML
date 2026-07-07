# Chapter 10 — PyTorch Practice Labs (from Linear Regression onward)

Prereqs you've already covered: tensors, autograd, GPU, in-place ops, gradient descent
loops from scratch. These labs pick up where that stops. Use the California housing
dataset (Chapter 9) for the regression labs and Fashion MNIST for the classifier labs.
Each section ends with a "go further" nudge — chase those if you have time.

---

## Lab 1 — Linear regression with raw tensors + autograd

- Load California housing, split into train/valid/test, and turn each split into float tensors.
- Normalize the features using tensor operations only (no `StandardScaler`). Compute the
  statistics on the training set, then apply them everywhere.
- The raw targets come out as 1D. Reshape them so predictions and targets are the same shape.
  Figure out which shape the math actually wants.
- Create a weight tensor and a bias tensor as trainable parameters. Initialize one randomly
  and the other at zero.
- Write a full batch gradient descent loop: predict, compute MSE, backprop, update under a
  no-grad context, reset gradients, print the loss each epoch.
- Use the trained parameters to predict a few new instances.

*Go further:* Sweep the learning rate across a few orders of magnitude — find where it
diverges, where it crawls, where it's just right. Does initializing the bias randomly
instead of at zero change anything here? Why might that change for a deeper network?

---

## Lab 2 — Same model, high-level API

- Replace your hand-rolled weight/bias with a single built-in linear layer. Inspect its
  weight and bias attributes. Compare the weight's shape to the tensor you made by hand in
  Lab 1 — they're related by a transpose. Make sure you can explain why.
- Iterate over the module's parameters two ways: anonymously, and as name/value pairs.
- Call the model on a small batch as if it were a function. Look at the result's `grad_fn`.
  What does calling the module actually trigger under the hood?
- Attach a hook that fires whenever the module runs, then remove it. Confirm it does *not*
  fire if you bypass the normal call path — and form an opinion on why you should never
  bypass it.
- Swap your manual update + zero-grad lines for an optimizer and a loss-function object.
  Wrap it all in a reusable training function.
- Train, predict, and compare numbers against Lab 1. They'll be close but not identical —
  track down the reason.

*Go further:* The loss object is also a "module." What else in this library turns out to be
a module? Why is that a useful design choice?

---

## Lab 3 — Regression MLP

- Stack a few linear layers with activations between them into one sequential model. Make the
  input width match your features and the output width match your targets, and make every
  layer's output line up with the next layer's input.
- Train it with the function you built in Lab 2.
- Compare its loss to the plain linear model.

*Go further:* Does adding more neurons or more layers keep helping, or does it plateau /
get worse? Try making all hidden layers the same width vs. different widths — does it matter?

---

## Lab 4 — Mini-batch gradient descent with DataLoaders

- Wrap your training tensors in a dataset object, then feed it through a loader that serves
  shuffled batches of a fixed size.
- Move the model to your device, and copy each batch to the device inside the loop.
- Write a `train()` function that: switches the model into training mode, loops epochs, loops
  batches, and reports the *mean* loss per epoch.
- Compare convergence and per-epoch wall-clock time against full-batch training.
- Explore the loader speed knobs one at a time and time the effect: pinned memory,
  non-blocking transfers, multiple worker processes, prefetching, persistent workers.

*Go further:* Skip the built-in dataset wrapper and write your own dataset class with just a
length method and an item-getter. What's the minimum interface a loader actually needs?
When do extra workers slow you down instead of speeding you up?

---

## Lab 5 — Evaluating a model

- Write an `evaluate()` function that takes a model, a loader, a per-batch metric function,
  and an aggregation function (defaulting to the mean). Run it in eval mode and without grad
  tracking.
- Compute validation MSE with it.
- Now compute RMSE by writing a per-batch root-mean-square-error and averaging the batches.
  Separately, take the square root of the MSE you already have. They won't match. Explain the
  discrepancy, then fix it by choosing the metric and aggregation functions so the math is
  correct over the whole set.
- Redo the RMSE using a streaming-metric library object. Confirm it agrees with your fixed
  version.

*Go further:* What does "streaming" buy you that recomputing from scratch doesn't? Extend
`train()` to record train and validation metrics every epoch and plot the learning curves.
What does the gap between the two curves tell you?

---

## Lab 6 — Custom modules and a nonsequential model

- Build a "wide & deep" network as a custom module subclass: a deep stack of layers plus a
  direct path from the input to the output layer. In the forward pass, concatenate the input
  with the deep stack's output before the final layer. Work out the output layer's input size.
- Train and evaluate it like your earlier models.
- Inspect the module's submodules (both anonymously and by name).
- Build a variant where the wide path and deep path each use a *different, possibly
  overlapping* subset of the input features, split inside the forward pass.

*Go further:* If a model held a variable number of sub-layers or parameters in a plain Python
list, they'd silently go missing from the parameter iterator. Find the container types meant
to hold them instead, and prove to yourself the plain-list version really does lose them.

---

## Lab 7 — Multiple inputs

- Change the wide & deep model so its forward pass takes two separate input tensors instead
  of one combined tensor.
- Build a dataset that returns the wide part, the deep part, and the target separately, and
  update your train/eval loops to move all of them to the device.
- Rewrite the loop so it works for *any* number of inputs, not just two — without naming them
  one by one.
- Then build a dataset that returns its inputs as a name→tensor mapping, and pass them to the
  model by name.

*Go further:* Why is naming inputs worth the extra ceremony once a model has many of them?
Construct a bug that the named version would have caught and the positional version wouldn't.

---

## Lab 8 — Multiple outputs

- Add a second ("auxiliary") output head that branches off the deep stack and predicts the
  target on its own.
- In training, compute a loss for each head and combine them into one number with a weighted
  sum. Tune the weight.
- In evaluation, ignore the auxiliary head entirely.

*Go further:* What's the auxiliary output actually *for* if you throw it away at eval time?
What would change if the two heads needed different targets and different loss functions?

---

## Lab 9 — Image classifier (Fashion MNIST)

- Load Fashion MNIST through the vision library, applying a transform pipeline that turns each
  image into a scaled float tensor.
- Inspect one sample: its shape has three dimensions now. Identify which is the channel
  dimension and where it sits. Why does this library put it there, and which libraries expect
  it elsewhere?
- Split off a validation set from the training data and build loaders for all three splits.
- Read the dataset's class-name list and map a sample's label to its name.
- Build a classifier as a custom module: flatten the image first, then a couple of
  hidden layers with activations, then an output layer with one unit per class and *no* final
  activation. Pair it with the multiclass cross-entropy loss.
- Train it, and evaluate accuracy with a streaming accuracy metric on the device. Compare
  train vs. validation accuracy and judge whether it's overfitting.
- Make predictions on a few images: turn logits into a predicted class, then into class names.
- Turn the logits into probabilities with a softmax. Then pull the top-k guesses and their
  probabilities for an image.

*Go further:*
- The model outputs logits, not probabilities — why does the loss prefer it that way?
- Add label smoothing to the loss and see what changes.
- Take an artificially imbalanced subset and set per-class loss weights so rare classes
  count more; work out how you'd normalize those weights.
- Deliberately remove the flatten step (or mangle a batch's shape) and read the error
  message carefully — shape errors are the most common beginner trap.
- Sketch (in your head or in code comments) how the loss/output setup would differ for a
  *binary* task and for a *multilabel* task.

---

## Lab 10 — Hyperparameter tuning with Optuna

- Write an objective function that asks the tuner for a learning rate (on a log scale) and a
  hidden-layer width, builds and trains the classifier with them, and returns validation
  accuracy.
- Run a small study that *maximizes* that score, with a fixed seed for reproducibility.
  Read back the best hyperparameters and best value.
- Refactor so the data loaders are passed into the objective explicitly instead of grabbed
  from global scope — using a lambda one time and a partial-application helper another time.
- Add a pruner: report the running validation accuracy after each epoch and abandon trials
  that look hopeless.

*Go further:* Why does a log-scale search find tiny learning rates that a uniform search
basically never would? Add more dimensions to the search (batch size, optimizer, layer count,
activation) and watch how fast the space blows up. When is the extra search worth it?

---

## Lab 11 — Saving and loading

- Save the whole trained model to disk, then load it back, switch it to eval mode, and
  predict with it.
- Now save *only* the learned weights. Reconstruct an identical architecture from scratch,
  load the weights into it, and predict. Articulate why the weights-only path is the safer,
  more portable one.
- Bundle the weights together with the hyperparameters needed to rebuild the architecture,
  save that, and reconstruct the model purely from the saved file.

*Go further:* What else would you need to save to *resume training* exactly where you stopped,
not just to run inference? Look into the safer weight-serialization format mentioned in the
chapter and what threat it's addressing.

---

## Lab 12 — Compiling and optimizing

- Convert your trained model to its compiled/serializable form by *tracing* it on a sample
  input. Save it, load it back in a fresh place, and run inference.
- Convert a model that contains an `if` or a loop in its forward pass by tracing, and inspect
  what got captured — then convert the same model by *scripting* and compare. Decide which
  method suits which kind of model.
- Apply an inference-time optimization pass and confirm the result is inference-only.
- Separately, wrap your model in the just-in-time compiler entry point, run it normally, and
  time inference before vs. after.

*Go further:* Tracing vs. scripting vs. JIT-compiling — what does each one actually capture,
and what breaks each one? Why would you ever ship the older serializable format instead of
just JIT-compiling at runtime?