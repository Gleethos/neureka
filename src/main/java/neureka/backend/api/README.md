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
operations for an equally large number of unique input tensor configurations... <br>
Additionally for every operation there might also 
be any number of concrete implementations tailored to specific
hardware, tensor dimensions or merely performance requirements.
Therefore, the API in this package is rather verbose,
nonetheless extremely powerful.

## Architecture ##

The package hosts a 3 tier layered API
made up of core concepts, namely: <br>

- **Operations** : A collection of species of algorithms.
- **Algorithms** : Representations of algorithms hosting multiple device specific implementations.
- **Implementations** : Implementations of an algorithm tailored to a specific `Device`.

This package models this architecture by exposing 3 interfaces which govern these concepts, 
and their relationship.
The interfaces are the following :

- `ImplementationFor<TargetDeice extends Device>`
- `Algorithm<ConcreteType>`
- `Operation`


Implementations of these interfaces are expected to form a composition / component based <br>
top to bottom structure.
This means that instances implementing the `ImplementationFor` interface are components <br>
of instances implementing the `Algorithm` interface <br>
which are themselves ultimately components of an `Operation` instance. <br>

## ExecutionCall ##

A key class within Neureka is the `ExecutionCall`. <br>
Instances of the class contain important context information for a <br>
given request for execution, just to name a view : <br>

- `Device` : The targeted device for the execution.

- `Operation` : The used operation type.

- `Tsr[]` : The tensor arguments for the operation.

- ... 

Instances of this class are being routed through this three tier <br>
architecture for final execution on instances of the 
`ImplementationFor<TargetDeice extends Device>` interface! <br>

## Algorithm ##

**Lets look into the interface methods and their meaning :** <br>

---

Every implementation has a name. <br>
This property is not always used, however when it comes to <br>
for example native code or dynamic kernel compilation the name is <br>
used to dynamically parse OpenCL code and identify it by said name. <br>

```java
    String getName();
```

**Note:** *The name should adhere to snake- or camel-case
as well as only contain letters, digits and underscores
so that it can be used as variable or method identifier
when using it for dynamic code compilation...*

---

When an `ExecutionCall` instance has been formed then it will be routed by <br>
the given `Operation` instance to their components, namely : <br>
`Algorithm` instances ! <br>

The ability to decide which algorithm is suitable for a given `ExecutionCall` instance <br>
is being granted by implementations of the following method. <br>
It **returns a float representing the suitability of a given call**. <br>
The float is expected to be between 0 and 1, where 0 means <br>
that the implementation is not suitable at all and 1 means that <br>
that it fits the call best! <br>

```java 
    float isSuitableFor( ExecutionCall<? extends Device<?>> call );
```
---


The following method ought to check if this
implementation can perform forward mode AD on
the given `ExecutionCall` instance.

```java
    boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call );
```
---

The following method ought to check if this
algorithm can perform backward mode AD on
the given `ExecutionCall` instance.

```java
    boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call );
```
---

This method ought to return a new instance
if the `ADAgent` class responsible for performing automatic differentiation
both for forward and backward mode differentiation. <br>
Therefore an `ADAgent` exposes 2 different procedures. <br>
One is the forward mode differentiation, and the other one <br>
is the backward mode differentiation which is more commonly known as back-propagation... <br>
Besides that it may also contain context information used <br>
to perform said procedures.

```java
    ADAgent supplyADAgentFor(
            neureka.calculus.Function f,
            ExecutionCall<? extends Device<?>> call,
            boolean forward
    );
```
---

The `ExecutionDispatcher` lambda
is the final execution procedure which is responsible for electing an `neureka.backend.api.ImplementationFor`
the chosen `Device` in a given `ExecutionCall`.
However, the  `ExecutionDispatcher` does not have to select a device specific implementation.
It can also occupy the rest of the execution without any other steps being taken.
For example, a `neureka.backend.api.ImplementationFor` or a `RecursiveExecutor`
would not be used if not explicitly called.
Bypassing other procedures is useful for full control and of course to implement unorthodox types of operations
like the `neureka.backend.standard.operations.other.Reshape` operation
which is very different from classical operations.
Although the `ExecutionCall` passed to implementations of this will contain
a fairly suitable `Device` assigned to a given `neureka.backend.api.Algorithm`,
one can simply ignore it and find a custom one which fits the contents of the given
`ExecutionCall` instance better.

```java
    Tsr<?> dispatch( Function caller, ExecutionCall<? extends Device<?>> call );
```

---

The execution call instance contains an array of arguments.<br>
Some of these arguments (usually the leading one(s)) are null
because they serve as output locations for the result of this `Algorithm`. <br>
The instantiation of these output tensors should be left to the
algorithm instance in most cases, this is because the given algorithm
"knows best" what shape(s), size(s), data type(s)... these tensors ought to have.<br>
<br>
An example would be algorithms performing element-wise operations vs broadcasting. <br>
The shape of an output tensor produced by an element-wise operation would most likely be
different from the shape of a tensor produced by broadcasting... <br>
<br>
This method ought to instantiate necessary output tensors:
```java
    ExecutionCall<? extends Device<?>> prepare( ExecutionCall<? extends Device<?>> call );
```

---

Implementations of the `Algorithm` interface ought to express a compositional design pattern. <br>
This means that concrete implementations of an algorithm for a device are not extending
an `Algorithm`, instead they are components of it. <br>
This design makes implementations both modular and highly extensible.

```java
    <D extends Device<?>, E extends ImplementationFor<D>> FinalType setImplementationFor( Class<D> deviceClass, E execution);
```
---

A device specific implementation can be accessed by passing the class of the implementation
of the `ImplementationFor<Device>` class.
An `Algorithm` instance ought to contain a collection of these device-specific
implementations...

```java
    <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass );
```

## ImplementationFor<Device> ##

This interface resides at the bottom most layer of
the 3 tier architecture of the backend API.
It is also the simplest of the three interfaces
because the implementation details are at this
point solely dependent on the device specific implementation. <br>
The `implementationFor<Device>` interface exposes only the following method:

```java
    void run( ExecutionCall<TargetDevice> call );
```

**What is this interface supposed to represent?**

Instances implementing the `ImplementationFor<TargetDevice extends Device>` interface <br>
represent the interaction between a given `ExecutionCall` instance and the targeted device. <br>
So the implementation describes this relationship by calling the device methods <br>
for a specific device implementation.  <br>
Here is a simplified example using opencl as backend : <br>

```java
// Custom executor for a given OperationTypeImplementation :

ImplementationFor<?> impl = 
            new CLImplementation( // implements 'ImplementationFor<OpenCLDevice>'
                (call) -> // A nested lambda containing the actual implementation...
                {
                   int gwz = call.getTsrOfType( Number.class, 0 ).size();
                   call.getDevice().getKernel(call)
                           .pass( call.getTsrOfType( Number.class, 0 ) )
                           .pass( call.getTsrOfType( Number.class, 1 ) )
                           .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                           .pass( call.getDerivativeIndex() ) 
                           .call( gwz );
               },
               3, // Arity (for correctness)
               someKernelSource, // kernelSource (maybe a file?)
               // activaion : 
               """
                    output = 
                       (
                            (float) log(
                                1 + pow(
                                    (float) M_E,
                                    (float) input
                                )
                            )
                        );
                """,
                // derivative :
                """
                    output =
                        1 /
                            ( 1 + (float) pow(
                                    (float) M_E,
                                    (float) input
                                )
                            );
                """,
                someOperation // Operation interface instance...
            );

```
