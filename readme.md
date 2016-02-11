Stupid Deep Net
====

Exactly like [stupid single-hidden-layer MLP](https://github.com/howonlee/stupidmlp), but... deeper. Also, of course, sparsification is now on all layers - and the order on which low magnitude params are killed off is defined over all the layers, too. The [Sustkever et al](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf) smart initialization was too rich for my blood, so the layers are just drawn from Gaussians with exponentially increasing standard deviation. You can get away with _even less_ burn-in to still get that fat tail on the weight histogram by cranking up the learning rate a lot just on those burn-in steps.
