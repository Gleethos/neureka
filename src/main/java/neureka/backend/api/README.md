
# The backend API #

## Introduction ##

Generally speaking there are three core
components which make up a minimal tensor library:

1. A densely packed and fancily indexed nd-array data structure marketed as "tensor".

2. A collection of algorithms which read and write to these tensors.

3. ...and some glue code which marries these two concepts cleanly together. 

The first point is a single implementation which is fairly simple.
Except for the indexing mechanism there is no need for complicated polymorphic concepts.
The same goes for the third point, which simply routes between the two layers...<br>
However, the second point is a bit tricky because
it involves a collection of multiple things which ought
to be called in some standardized way by the glue code. <br>
<br>
Therefore we need a pile of interfaces which allow
for polymorphic implementations.
That is exactly what an API is for! <br>

The package you are currently looking at is the 
definition for this standardized API.<br>
As you can image there is an infinitely large
number of different algorithms which could act as 
operations for an equally large number of input tensors... <br>
Additionally for every operation there might also 
be any number of concrete implementations tailored to specific
hardware, tensor dimensions or merely performance requirements.
Therefore, the API in this package is rather verbose,
nonetheless extremely powerful.

## Architecture ## 

The package hosts a 3 tier layered API
for Neureka's operation implementation backend.

For every layer of this architecture there is one package
in this `api` package, namely <br>

- `implementations` : Device specific implementations of an algorithm.

- `algorithms` : Representations of algorithms hosting multiple device specific implementations.

- `operations` : A collection of species of algorithms.

The architecture of the package consists of a composition / component based <br>
top to bottom structure where instances of the type defined in the 'implementations' package <br>
are components of those defines in the `algorithms` package <br>
which are themselves components of those defined in the 'operations' package. <br>

The 3 interfaces which govern this described relationship are the following :

- `ImplementationFor< TargetDeice extends Device >`

- `Algorithm< FinalType >`

- `Operation`

So to restate this relationship for clarity : <br>
Instances implementing the `ImplementationFor` interface are components <br>
of instances implementing the `Algorithm` interface <br>
which are themselves ultimately components of an Operation instance. <br>

## ExecutionCall ##

The architecture mainly deals with processing `ExecutionCall` instances. <br>
Instances of the class contain important context information for a <br>
given request for execution, namely : <br>

- `Device` : The targeted device for the execution.

- `Operation` : The used operation type.

- `Tsr[]` : The tensor arguments for the operation.

- ... 

Instances of this class are being routed through this three tier <br>
architecture for final execution on instances of the `ImplementationFor< TargetDeice extends Device >` interface! <br>



