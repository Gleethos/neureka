# Iterators #

This package contains a custom 
iterator interface type whose implementations 
are used
for efficient general purpose iterating
over the data which is stored in `Tensor`
instances...

Iterating over an ND-Array data structure
like they are present in tensors
requires incrementing an index array
whose length is the rank of the underlying tensor.
This index array is then being mapped 
to the real scalar index of the actual 
value somewhere inside the underlying data
array of a tensor. <br>

This process is in essence the access pattern of 
the data array, and it can be very simple or
very complex depending on the type of tensor (permuted, sliced, transposed...). 
Complicated access patterns require complicated 
implementations which are general purpose but less performant. 
Therefore, multiple case
specific `NDConfiguration` as well as a 
`NDIterator` implementations for tensors of
different dimensionalities make sense.
The purpose of the `NDIterator` interface specifically 
is to hide the optimized iteration logic underneath a
common interface which can then be used 
for iterating on any tensor efficiently.
<br>
This megamorphic
architecture makes inlining optimizations
for the JVM difficult, but after some warmup possible. 

## How do I use the NDIterator? ##

Take a look at this example:

```
public void iterateOver( Tensor t, int times ) 
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

The 'iter' object contains all relevant
context implementation for both an internal incremented index,
and a mapping to the "real index" used for data access.
Call the `i()` method to get the current data array index
and then call `increment()` in order to move the iterator to the next index.
There is also a `decrement()` method which moves back t the previous index.
 