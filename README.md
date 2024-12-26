# A ChatGPT clone, in 3000 bytes of C, backed by GPT-2

This program is a dependency-free implementation of GPT-2. It loads
the weight matrix and BPE file out of the original TensorFlow files,
tokenizes the input with a simple byte-pair encoder,
implements a basic linear algebra package with matrix math operations,
defines the transformer architecture, performs transformer inference,
and un-tokenizes the output with the BPE decoder.
All in ~3000 bytes of C.

It's optimized efficiently enough so that GPT-2 Small takes a few
seconds per reply on any modern machine. To do this I've implemented
KV caching and an efficient matrix multiplication algorithm,
with optional OMP parallelism.

You can then use this to create something like Chat GPT---just so long
as you don't care about the quality of the output. (It's actually
pretty terrible output, obejctively speaking... But it does run.)
There are a
few quirks (especially with handling UTF-8 characters), and running
the XL size model at long context length can require ~100GB of RAM.
But if you're just typing with ASCII using GPT2-Small it should run
just about anywhere.

# How does it work?

A complete description of how this code works is available at
https://nicholas.carlini.com/writing/2023/chat-gpt-2-in-c.html

# Running the code

First download the GPT-2 neural network

```
bash download.sh
```

First compile the code with, for example

```
gcc -O3 c_chat_gpt_2.c -lm
```

If you want to compile with OMP then you should pass -D GOFAST

```
gcc -O3 -D GOFAST c_chat_gpt_2.c -lm
```

Next you'll just want to start inference

```
bash run.sh
```

You should see something like

```
AI: How can I help you?
Human:
```

and from here you can just interact with it however you want. Remember though, this model is probably 1,000x smaller than GPT-3, who knows how much smaller than GPT-4, trained for probably thousands of times fewer steps, and is not fine-tuned to be a good chat model. So don't expect much. But it will run.


# LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.