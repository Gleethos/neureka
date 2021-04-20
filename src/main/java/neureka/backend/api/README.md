
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
made up of core concepts, namely: <br>

- `operations` : A collection of species of algorithms.

- `algorithms` : Representations of algorithms hosting multiple device specific implementations.

- `implementations` : Implementations of an algorithm tailored to a specific `Device`.

This package models this architecture by exposing 3 interfaces which govern these concepts, 
and their relationship.
The interfaces are the following :

- `ImplementationFor< TargetDeice extends Device >`

- `Algorithm< FinalType >`

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
`ImplementationFor< TargetDeice extends Device >` interface! <br>


## Algorithm ##

**Lets look into the interface methods and their meaning :** <br>

---

Every implementation has a name. <br>
This property is not always used, however when it comes to <br>
for example native code or dynamic kernel compilation the name is <br>
used to dynamically parse OpenCL code and identify it by said name. <br>

```
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

If the current algorithm has been chosen, then the target device <br>
for the execution is being picked by the method below. <br>

```java
    Device findDeviceFor( ExecutionCall<? extends Device<?>> call );
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
if the `ADAgent` class. <br>
This class contains field variables for 2 lambdas. <br>
One for th forward mode procedure and one <br>
for back-propagation... <br>
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

The following method is a bypass lambda used
for operations / algorithms that do not require a device specific execution.
The 2 methods following the one below will be ignored
if this returns a tensor. (a.k.a result) <br>
If the method returns 'null' then
the call routing will continue.

```java
    Tsr<?> handleInsteadOfDevice( AbstractFunction caller, ExecutionCall<? extends Device<?>> call );
```
---

Declared below is a call hook offering recursive
execution on calls with an arbitrary amount of arguments. <br>
A given implementation might only be able to handle a fixed amount
of tensor arguments, however this method solves this problem.<br>
The second argument is a lambda which can be called in order to
re-execute this very method! A new call instance with reduced (partially executed)
tensor arguments can be passed to the lambda.

```java
    Tsr<?> handleRecursivelyAccordingToArity(
                ExecutionCall<? extends Device<?>> call,
                Function<ExecutionCall<? extends Device<?>>,Tsr>goDeeperWith
            );
```
---

The execution call instance contains an array of arguments.<br>
Some of these arguments (usually the leading one(s)) are null
because they serve as output locations for this algorithm. <br>
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
    ExecutionCall<? extends Device<?>> instantiateNewTensorsForExecutionIn( ExecutionCall<? extends Device<?>> call );
```
---

The following method is used together with the `handleRecursivelyAccordingToArity` method.
There is already a great implementation of this method in the AbstractBaseAlgorithm class
which calls the 'handleRecursivelyAccordingToArity' method whose implementation
ought to either execute the given call which is passed to it or
continue to go deeper by calling the passed lambda... <br>

```java
    Tsr<?> recursiveReductionOf( ExecutionCall<? extends Device<?>> call, Consumer<ExecutionCall<? extends Device<?>>> finalExecution );
```
---

Implementations of the `Algorithm` interface ought to express a compositional design pattern. <br>
This means that concrete implementations of an algorithm for a device are not extending
an `Algorithm`, they are components of it instead. <br>


```java
    <D extends Device<?>, E extends ImplementationFor<D>> FinalType setImplementationFor(Class<D> deviceClass, E execution);
```
---

A device specific implementation can be accessed by passing the class of the implementation
of the `ImplementationFor<Device>` class.
An `Algorithm` instance ought to contain a collection of these device-specific
implementations...

```java
    <D extends Device<?>> ImplementationFor<D> getImplementationFor(Class<D> deviceClass );
```


## ImplementationFor<Device> ##

This interface represents the bottom most layer of
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
                   int gwz = call.getTensor( 0 ).size();
                   call.getDevice().getKernel(call)
                           .pass( call.getTensor( 0 ) )
                           .pass( call.getTensor( 1 ) )
                           .pass( call.getTensor( 0 ).rank() )
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
