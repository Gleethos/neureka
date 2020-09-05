
# The 'backend' package #

This package hosts a 3 tier abstraction architecture
for Neureka's operation implementation backend.

For every layer of this architecture there is one package
in the backend package, namely <br>

- executions

- implementations

- operations

The architecture of the 'backend' package consists of a composition / component based <br>
top to bottom structure where instances of the type defined in the 'execution' package <br>
are components of those defines in the 'implementations' package <br>
which are themselves components of those defined in the 'operations' package. <br>

The 3 interfaces which govern this described relationship are the following :

- execution.ExecutionFor< TargetDeice extends Device >

- implementations.OperationTypeImplementation< FinalType >

- operations.OperationType

So to restate this relationship for clarity : <br>
Instances implementing the ExecutionFor interface are components <br>
of instances implementing the OperationTypeImplementation interface <br>
which are themselves ultimately components of a OperationType instance. <br>

## ExecutionCall ##

The architecture mainly deals with ExecutionCall instances. <br>
Instances of the class contain important context information for a <br>
given request to execute, namely : <br>

- Device : The targeted device for the execution.

- OperationType : The used operation type.

- Tsr[] : The tensor arguments for the operation.

- ... 

Instances of this class are being routed through this three tier <br>
architecture for final execution on instances of the ExecutionFor class! <br>


