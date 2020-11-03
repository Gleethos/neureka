# The 'iterators' package #

This package contains a custom 
iterator interface type used
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
However this is not the only type of iteration done by Neureka.
Alternatively an array based iteration approach is
being performed for operations on tensors.


