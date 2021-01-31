
# Calculus #

This package hosts one of the most important 
components of this library, the Function interface alongside 
a backend responsible for instantiating its implementations. <br>
<br>
Generally speaking a tensor library has 3 core components: <br>

1. A densely packed and fancily indexed nd-array data structure marketed as "tensor".

2. A collection of algorithms which read and write to these tensors.

3. ...and some glue code which marries these two concepts cleanly together. 

This package is the **third one of these points**.
It glues together the `Tsr` class, and the backend API which 
consists of implementations of the `Operation` interface.
This makes sense conceptional because the mathematical 
foundation boils down two tensors and operations applied to them.
Operations can be combined by nesting them and abstracting them
as functions. <br>
This is the purpose of the `Function` interface as the name suggests.
Instances of this class **represent an abstract syntax tree
made up of tensor operations** which can be created dynamically during runtime.
<br>
One might wonder:

> why not interface with `Operation` instances directly?

The answer is simple: <br>
If a user of Neureka provides a String expression used to dynamically 
build a function AST wrapped by the `Function` interface enables a number
of optimizations which would otherwise not be possible. <br>
The function could for example be dynamically compiled into a 
new OpenCL kernel. The function could then be executed via
a single GPU call. Otherwise, Neureka would have to call 
the GPU every time for every operation the function consists of... <br>