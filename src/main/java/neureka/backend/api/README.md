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
**That is exactly what an API is for!** <br>

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

- `ImplementationFor<TargetDevice extends Device>`
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
given request for execution.
This context information is composed of the following: <br>

- `Tsr[]` : The tensor arguments for the operation.

- `Operation` : The used operation for the execution.

- `Device` : The targeted device for the execution.

- `Args` : Operation specific meta arguments. 

Once instantiated, the `ExecutionCall` will expose another
attribute chosen based on the above, namely it will choose a fitting
`Algorithm` instance among all algorithms in the targeted `Operation`!

Execution calls will be routed from an operation to a fitting algorithm<br>
and then be executed on an instance of the 
`ImplementationFor<TargetDeice extends Device>` interface!
(which is as mentioned earlier an algorithm component) <br>

## Algorithm ##

**Lets look into the interface methods and their meaning :** <br>

---

Every algorithm has a name. <br>
This property is not always used, however when it comes to <br>
for example native code or dynamic kernel compilation the name is <br>
used to dynamically parse OpenCL code and identify it by said name. <br>

```java
    String getName();
```

**Note:** *The name should adhere to snake- or camel-case
as well as only contain letters, digits and underscores
so that it can be used as variable or method identifier.
This is important for when using it for dynamic code compilation...*

---

When an `ExecutionCall` instance has been formed then it will be routed by
a given `Operation` instance to their components, namely : <br>
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


This method, defined by the `ADSupportPredicate` interface,
checks which auto differentiation mode
can be performed for a given `ExecutionCall`.
The method returns a `AutoDiffMode` enum instance.
This check is important so that the autograd system can apply 
a suitable autograd strategy...

```java
    AutoDiffMode autoDiffModeFrom( ExecutionCall<? extends Device<?>> call );
```
---

The method below ought to return a new instance
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

This method is defined by the `Execution` functional interface which
is supposed to be the final execution procedure responsible for electing an `neureka.backend.api.ImplementationFor`
the chosen `Device` in a given `ExecutionCall`.
However, the  `Execution` does not have to select a device specific implementation.
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
    Result execute(Function caller, ExecutionCall<? extends Device<?>> call );
```

---

An `ExecutionCall` instance contains an array of arguments.<br>
Some of these arguments (usually the leading one(s)) are null.
This is because they serve as output locations for the result of a given `Algorithm`. <br>
The instantiation of these output tensors should be left to the
algorithm instance in most cases. The reason for this is because a given algorithm
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

## ImplementationFor&#60;Device&#62; ##

This interface resides at the bottom most layer of
the 3 tier architecture of the backend API.
It is also the simplest of the three interfaces
because the implementation details are at this
point solely dependent on the device in use. <br>
The `implementationFor<Device>` interface exposes only the following method:

```java
    void run( ExecutionCall<TargetDevice> call );
```

**What is this interface supposed to represent?**

Instances implementing the `ImplementationFor<D extends Device>` interface <br>
represent the concrete implementation of a given `Algorithm` to which they belong, tailored to 
a specific `Device`.
They receive an `ExecutionCall` instance to perform the actual execution
requested by said call. This is also where the journey of an `ExecutionCall`
ought to finally come to an end. <br>
Here is a simplified example using opencl as backend : <br>

```java
// Custom executor for a given `ImplementationFor<OpencCLDevice>` :

ImplementationFor<OpenCLDevice> impl = 
                                    CLImplementation // implements 'ImplementationFor<OpenCLDevice>'
                                    .compiler()  
                                    .arity( 3 )
                                    .kernelSource( sourceText ) // kernelSource (maybe a file?)
                                    .activationSource( "output = input1 / input2;\n" )
                                    .differentiationSource( // This performs division
                                        "if (d==0) {\n" + // 'd' is the derivative index.
                                        "    output = 1/input2;\n" +
                                        "} else {\n" +
                                        "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                                        "}"
                                    )
                                    .kernelPostfix( this.getFunction() )
                                    .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                            .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                            .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                            .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                            .pass( call.getValOf( Arg.DerivIdx.class ) )
                                            .call( gwz );
                                        }
                                    )
                                    .build();

```
