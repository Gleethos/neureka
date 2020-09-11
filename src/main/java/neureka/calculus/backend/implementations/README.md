
# The 'implementations' package

This package is the middle layer of the 3 tier <br>
calculus backend architecture. <br>
Extending this layer is the most complicated of the three. <br> 
This is because such extensions / implementations 
can literally be any procedure done to an 'ExecutionCall' instance,
and the tensor arguments contained within this call. <br>

The core component of this layer is the <br>
'OperationTypeImplementation\<FinalType\>' interface! <br>

However when extending existing operations or creating new ones,   <br>
this interface should not be implemented directly. <br>
Instead there are 2 useful abstract classes which already <br>
implement the component logic expected by the interface. <br>

The referenced classes are :

- AbstractBaseOperationTypeImplementation

- AbstractFunctionalOperationTypeImplementation

The first abstract class implements a component system for 'ExecutionFor<TargetDevice>' <br>
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
the given 'OperationType' instance to their components, namely : <br> 
'OperationTypeImplementation' instances ! <br>

The ability to decide which implementation is suitable for a given 'ExecutionCall' instance <br>
is being granted by implementations of the following method. <br>
It returns a float representing the suitability of a given call. <br>
The float is expected to be between 0 and 1, where 0 means <br>
that the implementation is not suitable at all and 1 means that <br>
that it fits the call best! <br>

```
    float isImplementationSuitableFor( ExecutionCall call );
```
---

If the current implementation has been chosen, then the target device <br>
for the execution is being picked by the method below. <br>

```
    Device findDeviceFor( ExecutionCall call );
```
---

The following method ought to check if this 
implementation can perform forward mode AD on
the given 'ExecutionCall' instance.

```
    boolean canImplementationPerformForwardADFor( ExecutionCall call );
```
---


The following method ought to check if this 
implementation can perform backward mode AD on
the given 'ExecutionCall' instance.

```
    boolean canImplementationPerformBackwardADFor( ExecutionCall call );
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
for operations that do not require device execution.
The 2 methods following the one below will be ignores
if this returns a tensor. (a.k.a result) <br>
If the method return 'null' then 
the call routing will continue.

```
    Tsr handleInsteadOfDevice(  AbstractFunction caller, ExecutionCall call );
```
---

Declared below is a call hook offering recursive 
executions on calls with an arbitrary amount of arguments. <br>
A given implementation might only be able to handle a fixed amount 
of tensor arguments, however this method solves this problem.<br>


```
    Tsr handleRecursivelyAccordingToArity( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
```
---



```
    ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call );
```
---

The following method is used together with the 'handleRecursivelyAccordingToArity' method.

```
    Tsr recursiveReductionOf(ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );
```
---


```
    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);
```
---


```
    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);
```