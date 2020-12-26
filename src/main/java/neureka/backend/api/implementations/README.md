# The 'implementations' package #

This package is disappointingly empty because the concrete  <br>
implementations of this layer are (if no plugins or extensions are present)  <br> 
only located in the acceleration package.  <br>

**Why?** <br>

Well, this layer simply expects the implementation of the <br>
interface 'ImplementationFor\<TargetDevice extends Device\>'  <br>
which is as one might guess **device specific**, meaning  <br>
the context of the implementation is almost always much more <br>
dependent on the device backends than on anything  <br>
contained in the 'calculus' package ! <br>

So if you want to see examples of this, simply visit any  <br>
sub package of the 'backend.standard' package. <br>
It contains its own 'implementations' package hosting <br>
implementations of the 'ImplementationFor\<TargetDevice extends Device\>' class. <br>

**What is this interface supposed to represent?**

Instances implementing the 'ImplementationFor\<TargetDevice extends Device\>' interface <br>
represent the interaction between a given 'ExecutionCall' instance and the targeted device. <br>
So the implementation describes this relationship by calling the device methods <br>
for a specific device implementation.  <br>
Here is a simplified example using opencl as backend : <br>

(Groovy)
```
// Custom executor for a given OperationTypeImplementation :

def executor = new CLImplementation( // implements 'ImplementationFor<OpenCLDevice>'
    call -> // A nested lambda containing the actual implementation...
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
)


```