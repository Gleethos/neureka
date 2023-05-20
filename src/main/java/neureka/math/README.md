
# Calculus #

This package hosts one of the most important 
components of this library, the `Function` interface alongside its implementations. <br>
<br>
Generally speaking a tensor library has 3 core components: <br>

1. A densely packed and fancily indexed nd-array data structure marketed as "tensor".

2. A collection of algorithms which read and write to said tensors.

3. ...and some glue code which marries these two concepts cleanly together. 

This package is the **third one of these points**.
It glues together the `Tensor`/`Nda` API, and the backend API which 
consists of implementations of the `Operation` interface.
This makes sense conceptionally because the mathematical 
foundation boils down to tensors (numeric data) and a set of operations which may be applied to them.
Operations can be combined by nesting them and abstracting them
away as functions. <br>
This is the purpose of the `Function` interface as the name suggests.
Instances of this class **represent an abstract syntax tree
made up of tensor operations** which can be created dynamically during runtime.
<br>
One might wonder:

> *Why not interface with `Operation` instances directly?*

The answer is simple: <br>
Dynamically 
building a function AST wrapped by the `Function` interface, enables a number
of optimizations which would otherwise not be possible! <br>
The function could for example be dynamically compiled into a 
new OpenCL kernel. It can then be executed via
a single GPU call. Otherwise, Neureka would have to call 
the GPU every time for every operation the function consists of... <br>
