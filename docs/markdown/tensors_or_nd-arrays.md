
# Tensors or Nd-arrays #

*What is the difference?*

You may already have noticed that in the world of machine learning, 
we use something called a **'tensor'** to represent data.
They might be called **'nd-arrays'** in some other frameworks,
but although they are very similar, 
there are also some important distinctions to be made between these two concepts.
Both are at their core merely multidimensional arrays, however,
they are different in their typical usage and API.
Nd-arrays are merely used to represent any type of data as a 
collection of elements in a multidimensional grid,  
tensors on the other hand have additional requirements.
They are a type of nd-array which stores numeric data 
as well as expose various mathematical operations for said data.
In that sense it is actually merely a more complex kind of number.
This concept actually comes from the field of physics, 
where it is used to represent a physical quantity.

## Instantiating Them ##

Neureka models both concepts through the `Tsr` and the `Nda` interfaces.
`Nda` is an abbreviation of `NdArray`, and `Tsr` is an abbreviation of `Tensor`.
The `Tsr` type is a subtype of the `Nda` type, exposing additional methods
like for example `plus`, `minus`, `times` and `divide`.
Both can be instantiated through static factory methods (and a fluent builder API).

Here an example:
````java
// Tensor:
Tsr<Float>  t = Tsr.of(Float.class).withShape(2, 3).andFill(1f, 4f, -2f);
// NdArray:
Nda<String> s = Nda.of(String.class).withShape(2, 3).andFill("a", "b", "c");

// Tensors are ND-arrays, but NdArrays are not Tensors:
Nda<Float> f = t; // The other way around is not possible.
````


