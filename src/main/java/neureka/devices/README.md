
# Devices #

The core file of this package is the `Device` interface. <br>
Generally speaking devices are things that store tensors and
potentially also expose an API for operations make usage of. <br>

Because devices also store tensors the `Device` interface extends the `Storage`
interface. 
The `Device` interface also extends the `Component` interface.
This is because instances of Device implementations are also components of tensors.
Tensors *"need to know where they are located at"* while *"devices need to know which tensors they host"* ...
Therefore, the following two ways of marrying both types exist : <br>

```java
myTensor.to( myDevice );
```
... or ...
```java
myDevice.store( myTensor );
```

---

Instances of implementations of the `Device` interface can be accessed in different ways.
However, the simplest and safest way is the static `find(...)` method  located in the Device interface.
This works as follows : <br>

```java
var device = Device.find("cpu"); // "gpu", "first gpu", "amd", ...
```

