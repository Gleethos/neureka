# Extending Neureka #

Neureka is a lightweight library, meaning that its default backend operations only 
support `double` based tensors on the CPU and `float` based tensors on the GPU.
However, the existing backend of this library adheres to a standardized API
based on which you can easily extend it through a common library context!
<br>
 
First we need the following 3 little imports:
```java 
import neureka.Neureka;
import neureka.backend.api.BackendContext;
import neureka.backend.api.Operation;
```
Then we can start diving towards the backend
by first accessing the thread local library context:
```java   
Neureka library = Neureka.get();
```
This in turn contains and exposes the backend context
which contains operations and backend extensions!
```java   
BackendContext backendContext = library.backend();
```
Now we can build a custom backend extension and hook it into the default context:
```java   
if ( !backendContext.has(MyBackendExtension.class) )
    backendContext.set(new MyBackendExtension());
```
However, we don't have to use this, it's really just a common register for any kind of state. <br>
For example, this is used to register the OpenCL platforms and devices.

If we simply want to create a new type of operation we can just add them to the context like so:
```java  
if ( !backendContext.hasOperation("customFunction") )
    backendContext.addOperation(new MyCustomFunOperation());
```
Often times however, we simply want to extend an existing operation, like for example we might want to add
support for other kinds of data types for the `+`,`*`,`/`... operations! <br>
We can simply extend any given operation like so:
```java      
Operation plus = backendContext.getOperation("+");
plus.setAlgorithm(new MyAdditionAlgorithm());
``` 
For further details take a look at the `Algorithm` documentation
as well as the current implementations located in `neureka.backend.standard.*`
 
