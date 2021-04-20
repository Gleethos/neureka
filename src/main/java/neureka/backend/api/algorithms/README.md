
# Algorithms

This package expands the middle layer of the 3 tier 
calculus backend API architecture by partially implementing
the `Algorithm` interface. <br>
Implementing the `Algorithm` layer is the most complicated of the three. <br> 
This is because such extensions / implementations 
can literally be **any procedure done to an `ExecutionCall` instance**, 
and the tensor arguments contained within this call. <br>
Hence the package name "algorithms"! <br>

Consequently, this package contains some basic implementations
for the `Algorithm` interface! <br>

When extending existing operations or creating new ones,   <br>
this interface should almost never be implemented directly. <br>
Instead there are 2 useful abstract classes which already <br>
implement the component logic expected by the interface. <br> 

The referenced classes are :

- `AbstractBaseAlgorithm`

- `AbstractFunctionalAlgorithm`

- `GenericAlgorithm`

The first abstract class implements a component system for `ImplementationFor<TargetDevice>` <br>
instances, and the second class extends the first one and adds support for functional <br>
implementations of the overridable methods from the root interface. <br>
<br>
> *The latter of the two classes, namely `AbstractFunctionalAlgorithm`, should be 
> the preferred choice when building new custom operations.*
> *However, if the implementation of an entirely new high performance backend
> is required, then extending the `AbstractBaseAlgorithm` class would be ideal.*
