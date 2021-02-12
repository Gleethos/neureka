# The 'iterators' package #

This package contains a custom 
iterator interface type implementations used
for efficient general purpose iterating
over the data which is stored in 'Tsr'
instances...

Iterating over an ND-Array data structure
like they are present in tensors
requires incrementing an index array
whose length is the rank of the underlying tensor.
This index array is then being mapped 
to the real scalar index of the actual 
value somewhere inside the underlying data
array of a tensor. <br>

This process can be very simple as well as
very complex. There fore multiple case
specific implementations for tensors of
different dimensionalities make sense.
The purpose of the 'NDIterator' interface 
is to hide the underlying iteration underneath a
common interface which can then be used 
for iterating on any tensor efficiently.
<br>
Unfortunately this megamorphic
architecture hinders inlining optimizations
for the JVM. 
However, this is not the only type of iteration done by Neureka.
Alternatively an array based iteration approach is used 
for operations on tensors when performance
is important.<br>
The difference between the two approaches is being outlined below :

**NDIterator approach :** <br>
The 'iter' object contains all relevant
context implementation for both an internal incremented index, 
and a mapping to the "real index" used for data access.
```
public void iterateOver( Tsr t, int times ) 
{
    NDIterator iter = NDIterator.of( t );
    double[] tensorData = (double[]) t0_drn.getData();
    for ( int i = 0; i < times; i++ ) 
    {
        double current = tensorData[iter.i()];
        // ... used somehow ...
        // incrementing : 
        iter.increment();
    }
}
```

**Array based approach :**
This iteration method uses three components.
It uses two int arrays, one constant shape array,
and an index array which ought to be incremented according
to the max values inside the shape array.
The third component is an implementation instance of
the NDConfiguration interface which is responsible for
returning the "true scalar index" of the tensor data.
```
public void iterateOver( Tsr t, int times ) 
{
    NDConfiguration ndc = t.getNDConf();
    int[] tensorShape = ndc.shape();
    int[] arrayIndex = new int[ tensorShape.length ];
    double[] tensorData = (double[]) t0_drn.getData();
    for ( int i = 0; i < times; i++ ) 
    { 
        double current = tensorData[ ndc.indexOfIndices( arrayIndex ) ];
        // ... used somehow ...
        // incrementing :
        NDConfiguration.Utility.increment( arrayIndex, tensorShape );
        i++;
    }
}
```