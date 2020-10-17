
# The 'autograd' package #

This package hosts the classes responsible for enabling 
automatic differentiation and error propagation, namely "autograd".

In order to achieve this, occurring computations are being recorded in a computation graph. 
This graph is made up of GraphNode instances which store both references to parent and child nodes.

When calling variations of the `backward(..)` method on a tensor with tracked computation, 
then the GraphNode instances start recursively traversing the graph while also propagating
a given error to tensors which require gradients.

---

## When will a graph form exactly? ##

In order for computations to be tracked by the autograd system, three conditions must
be met : <br>

1. The used operation(s) must support auto-differentiation.

2. The used Function instance must not be "detached", meaning it was constructed with the "doAD" flag set to `true`.

3. The computation uses inputs which directly or indirectly reference a tensor which has its "requires autograd" flag set to `true`.

---










