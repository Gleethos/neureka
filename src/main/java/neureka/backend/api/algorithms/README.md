
# The 'algorithms' package

This package is the middle layer of the 3 tier <br>
calculus backend architecture. <br>
Extending this layer is the most complicated of the three. <br> 
This is because such extensions / implementations 
can literally be any procedure done to an 'ExecutionCall' instance,
and the tensor arguments contained within this call. <br>
Hence the package name "algorithms"! <br>

Consequently the core component of this layer is the <br>
'Algorithm\<FinalType\>' interface! <br>

However, when extending existing operations or creating new ones,   <br>
this interface should not be implemented directly. <br>
Instead there are 2 useful abstract classes which already <br>
implement the component logic expected by the interface. <br>

The referenced classes are :

- AbstractBaseAlgorithm

- AbstractFunctionalAlgorithm

The first abstract class implements a component system for 'ImplementationFor<TargetDevice>' <br>
instances, and the second class extends the first one and adds support for functional <br>
implementations of the overridable methods from the root interface. <br>

**Before going into further detail, <br>
lets look into these methods and their meaning :** <br>

---

Every implementation has a name. <br>
This property is not always used, however when it comes to <br>
for example native code or dynamic kernel compilation the name is <br>
used to dynamically parse OpenCL code and identify it by said name. <br>

```
    String getName();
```
---

When an 'ExecutionCall' instance has been formed then it will be routed by <br>
the given 'Operation' instance to their components, namely : <br> 
'Algorithm' instances ! <br>

The ability to decide which algorithm is suitable for a given 'ExecutionCall' instance <br>
is being granted by implementations of the following method. <br>
It returns a float representing the suitability of a given call. <br>
The float is expected to be between 0 and 1, where 0 means <br>
that the implementation is not suitable at all and 1 means that <br>
that it fits the call best! <br>

```
    float isAlgorithmSuitableFor( ExecutionCall call );
```
---

If the current algorithm has been chosen, then the target device <br>
for the execution is being picked by the method below. <br>

```
    Device findDeviceFor( ExecutionCall call );
```
---

The following method ought to check if this 
implementation can perform forward mode AD on
the given 'ExecutionCall' instance.

```
    boolean canAlgorithmPerformForwardADFor( ExecutionCall call );
```
---


The following method ought to check if this 
algorithm can perform backward mode AD on
the given 'ExecutionCall' instance.

```
    boolean canAlgorithmPerformBackwardADFor( ExecutionCall call );
```
---

This method ought to return a new instance 
if the ADAgent class. <br>
This class contains field variables for 2 lambdas. <br>
One for th forward mode procedure and one <br>
for backpropagation... <br>
Besides that it may also contain context information used <br>
to perform said procedures.

```
    ADAgent supplyADAgentFor(
            neureka.calculus.Function f,
            ExecutionCall<Device> call,
            boolean forward
    );
```
---

The following method is a bypass lambda used 
for operations / algorithms that do not require device specific execution.
The 2 methods following the one below will be ignored
if this returns a tensor. (a.k.a result) <br>
If the method returns 'null' then 
the call routing will continue.

```
    Tsr handleInsteadOfDevice(  AbstractFunction caller, ExecutionCall call );
```
---

Declared below is a call hook offering recursive 
execution on calls with an arbitrary amount of arguments. <br>
A given implementation might only be able to handle a fixed amount 
of tensor arguments, however this method solves this problem.<br>
The second argument is a lambda which can be called in order to 
re-execute this very method! A new call instance with reduced (partially executed)
tensor arguments can be passed to the lambda.

```
    Tsr handleRecursivelyAccordingToArity( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
```
---

The execution call instance contains an array of arguments.<br>
Some of these arguments (usually the leading one(s)) are null
because they serve as output locations for this algorithm. <br>
The instantiation of these output tensors should be left to the
algorithm instance in most cases, this is because the given algorithm
"knows best" what shape(s), size(s), data type(s)... these tensors ought to have.<br>
<br>
An example would be algorithms performing elementwise operations vs broadcasting. <br>
The shape of an output tensor produced by an elementwise operation would most likely be
different from the shape of a tensor produced by broadcasting... <br>
<br>
This method ought to instantiate necessary output tensors:
```
    ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call );
```
---

The following method is used together with the 'handleRecursivelyAccordingToArity' method.
There is already a great implementation of this method in the AbstractBaseAlgorithm class
which calls the 'handleRecursivelyAccordingToArity' method whose implementation
ought to either execute the given call which is passed to it or
continue to go deeper by calling the passed lambda... <br>

```
    Tsr recursiveReductionOf( ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );
```
---

Implementations of the Algorithm interface ought to express a compositional design pattern. <br>
This means that concrete implementations of an algorithm for a device are not extending
an Algorithm, they are components of it instead. <br>


```
    <D extends Device, I extends ImplementationFor<D>> FinalType setImplementation( Class<I> deviceClass, I execution );
```
---

A device specific implementation can be accessed by passing the class of the implementation 
of the 'ImplementationFor<Device>' class.
An Algorithm instance ought to contain a collection of these device specific 
implementations...

```
    <D extends Device, I extends ImplementationFor<D>> ImplementationFor getImplementation( Class<I> deviceClass );
```