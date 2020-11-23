
# The 'devices' package #

The core file of this package is the "Device" interface. <br>
Generally speaking devices are things that store tensors and
optionally also have implementations for operations associated with
them in the calculus package. <br>

Because devices also store tensors the "Device" interface extends the "Storage"
interface. 
The Device interface also extends the "Component" interface located inside the main package,
namely the "neureka" package.
This is because instances of Device implementations are also components of tensors.
Tensors "need to know were they are located at" while "devices need to know which tensors they host" ...
Therefore the following two ways of marrying both types exist : <br>

```
myTensor.set( myDevice );
```
... or ...
```
myDevice.store( myTensor );
```

---