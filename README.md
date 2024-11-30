
# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

## ng cache progress

Problem: caching takes longer, I think is because the saving of computation time is less than caching time. In a large model, this will be resolved. So the problem is that I don't know how to cusotimze cacheing in a huggingface model.

Finished:

- One Head, One Layer, One input token, k_cache and v_cache seperated
- Head has an input "cache" which inlude both k_cache and v_cache
- mutliple Heads
- multiple layer
Doing:

- multiple Head & multiple Layer

Todo:

- Multiple input tokens
