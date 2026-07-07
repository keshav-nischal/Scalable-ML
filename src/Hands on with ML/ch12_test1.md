# Chapter 12 — Deep Computer Vision with CNNs: Lab Guide

A sequence of hands-on labs covering every worked example in the chapter. Each lab gives you a **goal**, a **loose task** (figure out the exact API calls yourself), and a few **things to wonder about** so you walk away with more questions than you started with.

Work top to bottom — later labs reuse code and intuitions from earlier ones. Use a GPU if you have one, but most early labs run fine on CPU with small inputs.

---

## Part 1 — Building Blocks from First Principles

### Lab 1.1 — The cost of going fully connected
**Goal:** feel *why* CNNs exist before you write one.

**Task:** Write a function that, given an image size and a number of neurons in the first layer, returns the number of weights a fully connected layer would need. Plot how that number grows as the image goes from tiny (Fashion-MNIST scale) to a modest photo (a few hundred pixels per side). Then compute the same thing for a convolutional layer with a handful of small filters.

**Wonder about:**
- At what image size does the fully connected count become absurd?
- The conv count barely moves when the image grows. Which property of conv layers makes that true, and what does it cost you in return?

### Lab 1.2 — Hand-built filters and feature maps
**Goal:** see that a filter is just a small image of weights.

**Task:** Load one or two sample images and convert them to a float tensor in the layout PyTorch expects (mind where the channel dimension goes). Build a vertical-line filter and a horizontal-line filter *by hand* as small matrices, and apply them across the image to produce two feature maps. Display the inputs and outputs side by side.

**Wonder about:**
- Why do the white lines get *enhanced* while everything else blurs?
- What happens at the very edges of the output, and why?
- If you rotated your filter 45°, what would the feature map highlight?

### Lab 1.3 — The convolution equation, by hand
**Goal:** demystify the scary multi-index equation.

**Task:** Take a small input (say a single 5×5×1 patch), a known kernel, a stride, and a bias. Compute one output neuron's value using nested loops, following the equation literally. Then feed the same input through a real conv layer with the same weights and confirm you get the same number.

**Wonder about:**
- Which indices in the equation correspond to *moving the window* versus *moving inside the window*?
- Where exactly does stride enter, and how does it differ from kernel size?

---

## Part 2 — Convolutional Layers in PyTorch

### Lab 2.1 — Your first real conv layer
**Goal:** map the math onto the framework.

**Task:** Create a 2D conv layer with several output channels and a moderate kernel size. Feed it a small batch of images and inspect the output shape. Predict the output height/width *before* running it, then check.

**Wonder about:**
- The input is 4D but the layer is called "2D." What do the two extra dimensions mean?
- The output shrank. By exactly how many pixels, and how does that relate to kernel size?

### Lab 2.2 — Padding and stride as shape controls
**Goal:** learn to steer output size deliberately.

**Task:** Run the same layer with "valid" padding, then "same" padding, and confirm the output sizes match your predictions. Then introduce a stride greater than 1 and see the spatial dimensions collapse. Try to find a padding value that keeps output size equal to input size *with* a stride — and notice what the framework does about it.

**Wonder about:**
- Why is zero-padding called "same," and why is no padding confusingly called "valid"?
- Why might setting a huge padding value to force equal sizes be a bad idea?

### Lab 2.3 — Peeking inside the layer
**Goal:** connect weights/biases to the equation's symbols.

**Task:** Inspect the weight and bias tensors of a conv layer and explain each dimension out loud. Confirm the kernel shape is independent of input image size. Re-initialize the weights using an initializer suited to ReLU, and zero the biases.

**Wonder about:**
- Why doesn't the input image's height/width appear anywhere in the weight shape?
- What breaks if you stack two conv layers with *no* activation between them?

---

## Part 3 — Pooling

### Lab 3.1 — Max pooling and what it throws away
**Goal:** subsample with intent.

**Task:** Build a max pooling layer and apply it to an image. Verify how much the spatial size shrank and estimate what fraction of values were discarded. Swap in average pooling and compare the two outputs visually on the same image.

**Wonder about:**
- Max pooling deletes most of its input yet often *helps*. Why might keeping only the strongest response be a feature, not a bug?
- When would average pooling preserve something max pooling destroys?

### Lab 3.2 — Translation invariance experiment
**Goal:** measure invariance instead of taking it on faith.

**Task:** Make a simple binary "image" with a shape in it. Create shifted copies (one pixel, two pixels). Pass all of them through max pooling and compare outputs. Quantify how much the output changes per pixel of shift.

**Wonder about:**
- For which shifts is the output identical, and for which does it move? Can you predict this from the pooling stride?
- For a task like *segmentation*, would this invariance be a help or a problem? What property would you want instead?

### Lab 3.3 — Pooling along an unusual axis
**Goal:** generalize pooling beyond height/width.

**Task:** Implement a pooling operation that aggregates across the *channel* dimension rather than the spatial one. Feed it a batch with many channels and verify the output channel count shrinks while spatial size stays put. (Hint: you'll need to rearrange dimensions, pool along one axis, and rearrange back.)

**Wonder about:**
- What kind of invariance could a network learn if you pool across channels — and why would you ever want it?
- Why doesn't a stock spatial-pooling layer give you this for free?

### Lab 3.4 — Global average pooling
**Goal:** collapse a feature map to a single number.

**Task:** Reduce each feature map of a batch to one value per map, using two different approaches (an adaptive pooling layer, and a plain reduction over the spatial axes). Confirm both give the same result.

**Wonder about:**
- This is extremely lossy. Why is it nonetheless a sensible thing to do *right before a classifier*?
- What does the adaptive version free you from having to know in advance?

---

## Part 4 — Assembling a CNN

### Lab 4.1 — A CNN from the standard recipe
**Goal:** build the conv → ReLU → pool → repeat → dense stack.

**Task:** Construct a small classification CNN for a grayscale dataset. Use a helper to avoid repeating the same conv arguments. Double the filter count after each pooling stage. End with a flatten, a couple of dense layers with dropout, and a logits output.

**Wonder about:**
- Why double the filters every time you halve the spatial size? What stays roughly balanced when you do?
- Why leave the softmax *out* of the model and fold it into the loss instead?

### Lab 4.2 — Finding the flatten dimension the hard way
**Goal:** stop guessing the first dense layer's input size.

**Task:** Deliberately set the first dense layer's input features to a wrong number and let training crash. Read the size out of the error message. Then re-derive that same number on paper by tracking the spatial size through each pooling layer. Finally, replace the layer with a "lazy" variant that figures it out automatically.

**Wonder about:**
- Why does the lazy layer need to see one batch before it can size itself?
- Could you compute the flatten size with a single dummy forward pass instead of crashing? Try it.

### Lab 4.3 — Train it and read the scoreboard
**Goal:** get a real accuracy number.

**Task:** Train your CNN on the dataset and evaluate on the test set. Note the accuracy and compare it to what a plain dense network would get on the same data.

**Wonder about:**
- Where is the model spending its parameters — the conv stack or the dense head? Count both.
- What's the single cheapest change you could make to push accuracy higher?

---

## Part 5 — Classic Architectures (reimplement from their tables/diagrams)

### Lab 5.1 — LeNet-5 from the spec
**Goal:** translate an architecture table into code.

**Task:** Build LeNet-5 layer by layer from its description (conv/pool/dense stack, the historical activations). Then make a "modernized" version swapping the old activations for current defaults. Compare parameter counts.

**Wonder about:**
- Which of LeNet-5's choices were essential ideas, and which were just artifacts of 1998?

### Lab 5.2 — AlexNet's shape bookkeeping
**Goal:** track tensor shapes through a deeper net.

**Task:** Implement AlexNet's layer sequence from its table. Before running, predict the spatial size after each layer and verify with a dummy forward pass. Add the two regularization ideas it used (a dropout scheme on the dense layers, plus data augmentation — see Lab 5.3).

**Wonder about:**
- AlexNet stacked conv layers directly without a pool between every one. Why was that a notable change?
- One of its normalization tricks fell out of fashion. What replaced it, and why?

### Lab 5.3 — Data augmentation as regularization
**Goal:** manufacture realistic training variety.

**Task:** Build an augmentation pipeline that flips, rotates, resizes/crops, and jitters colors. Apply it to a handful of images and display the variants. Make sure a human still recognizes each as the original object.

**Wonder about:**
- Why does *learnable* variation (shifts, flips) help, while pure random noise doesn't?
- How would you use augmentation to fix a class-imbalanced dataset?
- Sketch how you'd average predictions over several augmented copies of a *test* image. When is that worth the extra compute?

### Lab 5.4 — The inception module
**Goal:** run several kernel sizes in parallel and merge them.

**Task:** Build one inception module: parallel branches with different kernel sizes plus a pooling branch, all preserving spatial size, concatenated along depth. Insert 1×1 conv "bottlenecks" before the expensive branches.

**Wonder about:**
- A 1×1 conv looks at one pixel at a time — so what *can* it actually learn?
- How do the bottlenecks cut parameter count while *keeping* the same output spatial size?
- Why must every branch use the same padding/stride for the concatenation to work?

### Lab 5.5 — Residual learning
**Goal:** make a block model the *difference* from its input.

**Task:** Build a residual unit: a couple of conv+norm+activation layers whose output is *added* to the unit's input before a final activation. Handle the case where the block changes channel count or spatial size, so the skip path has to reshape the input to match.

**Wonder about:**
- At initialization, a residual block behaves almost like the identity. Why does that make very deep networks trainable?
- When the main path halves the spatial size, what exactly do you do to the skip path so the addition is even legal?

### Lab 5.6 — Depthwise separable convolution
**Goal:** split spatial and cross-channel mixing.

**Task:** Implement a separable conv as two stages: a depthwise stage (one spatial filter per input channel) followed by a 1×1 pointwise stage that mixes channels. Use the grouping argument of the conv layer to get the depthwise behavior. Compare its parameter count against a regular conv producing the same output shape.

**Wonder about:**
- What assumption about images does this layer *bet on*? When might that bet be wrong?
- Why is it a bad idea to use this right after a layer with very few channels (like the input)?
- Where does an inception module sit on the spectrum between a regular conv and a separable conv?

### Lab 5.7 — The squeeze-and-excitation block
**Goal:** let the network recalibrate its own feature maps.

**Task:** Build an SE block: global-average-pool each feature map to one number, squeeze that vector down through a small bottleneck dense layer, expand it back up to one weight per feature map (bounded between 0 and 1), then multiply each feature map by its weight. Attach it to the output of a residual unit or inception module.

**Wonder about:**
- The squeeze layer is deliberately tiny. What is that bottleneck *forcing* the block to learn?
- The block ignores spatial position entirely. Why is "which features fire together" still useful without knowing *where*?

---

## Part 6 — Architecture Trade-offs

### Lab 6.1 — Reading the pretrained-model comparison table
**Goal:** reason about accuracy vs size vs compute.

**Task:** Pull the list of available pretrained classification models from the vision library along with their reported accuracy, parameter count, and compute cost. Make a scatter plot of accuracy against parameters, and another against compute.

**Wonder about:**
- Find a case where a *smaller* model beats a larger one. What does that tell you about "bigger = better"?
- If you had to deploy to a phone, which axis would you optimize, and which model would you pick?

### Lab 6.2 — Compound scaling on paper
**Goal:** understand how to grow a network "in proportion."

**Task:** Given the constraint that depth, width, and resolution scale as powers of three coefficients (with the product of width-squared and resolution-squared and depth held near a target), write code that, for a chosen compute-budget exponent, outputs the suggested depth/width/resolution multipliers. Reproduce roughly the published baseline coefficients.

**Wonder about:**
- Why scale all three dimensions together instead of just stacking more layers?
- If your compute budget doubles, by how much should each dimension grow?

### Lab 6.3 — Memory accounting: inference vs training
**Goal:** predict when you'll run out of GPU RAM.

**Task:** For a single conv layer (pick filters, kernel size, input size), compute by hand: its parameter count, the number of multiply operations for one image, and the memory its output activations occupy for one image and for a batch. Then explain why training needs far more memory than inference for the *same* network.

**Wonder about:**
- During inference you can free a layer's memory once the next layer is done. Why can't you do that during training?
- List five different levers you could pull if training crashes with out-of-memory. Rank them by how much they hurt accuracy.

### Lab 6.4 — Trading compute for memory: checkpointing
**Goal:** recompute activations instead of storing them.

**Task:** Take a module inside a model and wrap it so its activations are *not* saved during the forward pass but recomputed during the backward pass. Confirm the model still trains and that predictions are unchanged at inference time.

**Wonder about:**
- The wrapped function must give the same output if called twice with the same input. Which common operations could quietly violate that, and why would it corrupt your gradients?

### Lab 6.5 — A reversible layer (stretch)
**Goal:** store *no* activations at all.

**Task:** Implement a reversible layer that takes two equal-sized inputs and produces two outputs via the additive coupling rule. Then write the inverse that recovers the inputs from the outputs. Verify numerically that forward-then-inverse round-trips exactly.

**Wonder about:**
- Why can't a reversible layer contain a stride>1 or "valid"-padding conv? What would that break about reversibility?
- The very first layer of a CNN usually *does* downsample. How would you feed its output into the first reversible layer?

---

## Part 7 — Implementing and Using ResNet

### Lab 7.1 — ResNet-34 from scratch
**Goal:** assemble a competition-winning net in ~one screen of code.

**Task:** Using your residual unit from Lab 5.5, build the full ResNet-34: a large-kernel strided stem, a max pool, then a stack of residual units following the "how many units at each filter count" recipe, ending in global average pooling and a linear classifier. Write the loop that decides each unit's stride based on whether its filter count changed from the previous unit.

**Wonder about:**
- Why is the stride set to 2 exactly when the filter count increases, and 1 otherwise?
- ResNet-152 uses a different *three-layer* residual unit. What problem does that bottleneck design solve at great depth?

### Lab 7.2 — Standing on the shoulders of pretrained models
**Goal:** classify real images with two lines of model loading.

**Task:** Load a modern pretrained classifier with its ImageNet weights. Preprocess your sample images using the transforms that *ship with those weights* (don't hand-roll the resize/normalize). Put the model in evaluation mode, disable gradient tracking, and get predictions. Map the predicted class IDs to human-readable names, and look at the top-3 with their probabilities.

**Wonder about:**
- Why is it safer to use the weights' own transforms than to write your own resize + normalize?
- The model is in training mode by default. Which layers behave differently in eval mode, and how would forgetting to switch corrupt your results?
- Your image isn't an exact ImageNet class. Are the top guesses *reasonable* substitutes? What does that say about what the features encode?

---

## Part 8 — Transfer Learning

### Lab 8.1 — Swap the head, freeze the body
**Goal:** adapt a pretrained net to new classes with little data.

**Task:** Take the pretrained model from Lab 7.2 and a small fine-grained dataset (far fewer images per class than you'd want to train from scratch). Inspect the model's named submodules to locate its classification head, replace the final layer to output the right number of classes, freeze every parameter, then unfreeze only the new head. Train a few epochs and record accuracy.

**Wonder about:**
- How can you reach high accuracy while training only a tiny fraction of the parameters?
- How did you *find* the layer to replace without reading the docs? Could you have done it programmatically?

### Lab 8.2 — Unfreeze and fine-tune
**Goal:** squeeze out more accuracy.

**Task:** After the head has trained, unfreeze the whole network, drop the learning rate substantially, and continue training. Then add an augmentation pipeline (remembering it must run *before* the normalization step) and try again.

**Wonder about:**
- Why lower the learning rate the moment you unfreeze the pretrained layers?
- Try giving lower layers a smaller learning rate than upper layers. Why might "differential learning rates" beat one global rate?
- Your dataset is unusual (say medical or satellite imagery). Why might ImageNet features *not* transfer well, and what would you look for instead?

---

## Part 9 — Localization

### Lab 9.1 — Add a bounding-box head
**Goal:** predict *where* an object is, not just *what*.

**Task:** Wrap your pretrained base model so it has two heads sharing the same features: the existing classification head, and a new head that outputs four numbers for a bounding box. Run a batch through and confirm you get both outputs per image.

**Wonder about:**
- What are sensible meanings for those four numbers? Name at least two different box parameterizations.
- You'd train this with a classification loss plus a regression loss. How do you combine two losses into one, and what could go wrong if you weight them badly?

### Lab 9.2 — Working with bounding boxes as tensors
**Goal:** transform boxes alongside images.

**Task:** Create a bounding box object in a center-based format with a known canvas size. Pass it through the *same* augmentation/preprocess transform you apply to images and observe how the coordinates change. Convert between box formats and visualize a box drawn on its image.

**Wonder about:**
- When you rotate an image, the box can't actually rotate. What does the transform do instead, and why does the box end up slightly too big?
- A 10-pixel error on a tiny box matters more than on a huge box, yet plain squared-error treats them equally. How would you reshape the targets to fix that?

### Lab 9.3 — Measuring box quality with IoU and friends
**Goal:** evaluate localization properly.

**Task:** Implement intersection-over-union for two boxes from scratch and check it against the library function. Then read about and use the generalized and complete IoU losses.

**Wonder about:**
- Plain IoU is zero whenever boxes don't overlap — and so is its gradient. Why does that make it useless as a *training* signal, and what extra information does generalized IoU add to fix it?
- Complete IoU adds two more geometric terms. What are they, and intuitively why would each one pull a predicted box toward the target faster?

---

## Part 10 — Object Detection

### Lab 10.1 — Sliding a classifier across a grid (conceptual sim)
**Goal:** understand the naive detection approach.

**Task:** Simulate chopping an image into a grid and "sliding" a fixed-size window across it, recording for each position a class guess, a box, and an *objectness* score. You don't need a trained detector — fabricate plausible scores — the point is the bookkeeping. Count how many predictions you end up with.

**Wonder about:**
- Why predict objectness *separately* from class, instead of adding a "no object" class?
- The same object gets detected several times at nearby positions. What post-processing removes the duplicates?

### Lab 10.2 — Non-max suppression
**Goal:** prune overlapping detections.

**Task:** Given a list of boxes with objectness scores, implement NMS: drop low-confidence boxes, then repeatedly keep the highest-scoring box and discard others that overlap it too much. Use your IoU function from Lab 9.3 for the overlap test.

**Wonder about:**
- How does the overlap threshold change behavior? What happens if it's too high or too low?
- Two genuinely different objects sit very close together. How might NMS wrongly delete one, and how would you guard against it?

### Lab 10.3 — Dense layer → conv layer equivalence
**Goal:** turn a classifier into something that scans any-size images.

**Task:** Take a dense layer sitting on top of a feature map and replace it with a conv layer that produces *numerically identical* outputs (match the filter count to the unit count, the kernel size to the feature-map size, with "valid" padding). Verify the two produce the same numbers (you can even copy the weights across). Then feed a *larger* image through the conv version and watch it emit a grid of predictions.

**Wonder about:**
- Why can a conv layer accept any input size while the dense layer it replaced cannot?
- If a 224-size image yields a single prediction, what grid size would a 448-size image yield, and why? Derive it from the network's overall stride.
- This "look once across the whole image" idea is the seed of a famous detector family. Which one?

### Lab 10.4 — Off-the-shelf detection and mAP
**Goal:** run a real detector and read its metric.

**Task:** Use a ready-made detection library to load a pretrained model and detect objects in a couple of images (it should accept URLs or arrays). Inspect the per-object output: class, confidence, box. Separately, read up on mean average precision and explain, in your own words, why it's a "mean of means."

**Wonder about:**
- Why summarize a precision/recall curve by *maximum precision at each recall level* rather than raw precision?
- What does mAP@0.5 mean, and how does the COCO-style averaged mAP differ from it? Why average over several IoU thresholds at all?

---

## Part 11 — Tracking, Segmentation, and Odds & Ends

### Lab 11.1 — Object tracking on video
**Goal:** keep identities consistent across frames.

**Task:** Use the same detection library's tracking mode on a short video. For each frame, print the track IDs of detected objects and save an annotated copy of the video.

**Wonder about:**
- A tracker combines a motion predictor, an appearance/resemblance model, and an assignment step. Picture two similar objects crossing paths — which component prevents their IDs from swapping, and which one would fail alone?
- Why does assuming "objects move at constant velocity" both help *and* occasionally mislead (e.g. a sudden bounce)?

### Lab 11.2 — Upsampling with a transposed convolution
**Goal:** grow a feature map back up to image size.

**Task:** Build a transposed conv layer and use it to enlarge a small feature map. Confirm that here the *stride controls how much the input is stretched*, so a bigger stride yields a bigger output — the opposite of a normal conv. Initialize it to approximate simple interpolation, then note that it's trainable and can learn to do better.

**Wonder about:**
- People sometimes call this a "deconvolution." Why is that name misleading?
- Why does pure interpolation (no learning) stop working well past a small upscaling factor?

### Lab 11.3 — Sketch a fully convolutional segmenter
**Goal:** classify every pixel, recovering lost resolution.

**Task:** On paper or in code-stubs, lay out the FCN segmentation idea: a pretrained CNN backbone (overall stride ~32) turned fully convolutional, then upsampling back to input size. Then add skip connections that *add in* outputs from lower, higher-resolution layers before further upsampling, following the upsample-add-upsample-add-upsample pattern.

**Wonder about:**
- The backbone applies an overall stride of 32. Why is upsampling by 32 in one shot "too coarse," and how do skip connections from lower layers sharpen the result?
- Segmentation wants *equivariance* (shift the input, the output shifts too), but pooling gave you invariance earlier. How does this architecture reconcile that tension?
- How is *instance* segmentation a harder problem than *semantic* segmentation, and what does a mask-producing detector add to solve it?

### Lab 11.4 — The other convolution flavors
**Goal:** know your toolbox beyond 2D conv.

**Task:** Build a tiny example each for a 1D conv (on a sequence) and a 3D conv (on a small volume), confirming the output shapes. Then take an ordinary 2D conv and crank up its *dilation*; verify the receptive field grows while the parameter count stays the same.

**Wonder about:**
- A dilated ("à-trous") filter inserts holes. How does that buy you a larger receptive field "for free," and what might you *lose* by skipping pixels?
- For what data shapes would you reach for 1D vs 3D conv? Name a concrete use of each.

---

## How to use this guide

- Don't peek at the chapter's exact code until you've struggled with the loose task for a bit — the struggle is where the learning is.
- After each lab, write one sentence answering its "wonder about" prompts. If you can't, that's your next thing to read.
- Labs marked or phrased as "conceptual" / "on paper" don't need a full training run — a dummy forward pass or a notebook sketch is enough to lock in the idea.