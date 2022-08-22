package ut.introductions

import neureka.Nda
import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensors or Nd-arrays")
@Narrative('''

*What is the difference?*

In the world of machine learning we use something called a **'tensor'** to represent data.
They might be called **'nd-arrays'** in some other frameworks,
but although they are very similar, 
there are also some important distinctions to be made between these two concepts.
Both are at their core merely multidimensional arrays, however,
they are different in their typical usage and API.
nd-arrays are merely used to represent any type of data as a 
collection of elements in a multidimensional grid,  
tensors on the other hand have additional requirements.
They are a type of nd-array which stores numeric data 
as well as expose various mathematical operations for said data.
In that sense it is actually merely a more complex kind of number.
This concept actually comes from the field of physics, 
where it is used to represent a physical quantity.

Neureka models both concepts through the `Tsr` and the `Nda` interfaces.
`Nda` is an abbreviation of `NdArray`, and `Tsr` is an abbreviation of `Tensor`.
The `Tsr` type is a subtype of the `Nda` type, exposing additional methods
like for example `plus`, `minus`, `times` and `divide`.
Both can be instantiated through static factory methods (and a fluent builder API).

''')
@Subject([Nda, Tsr])
class Tensor_NDArray_Spec extends Specification
{
    def 'Tensor is a subtype of NdArray.'()
    {
        given : 'A tensor of floats and an nd-array of strings:'
            Tsr<Float> t = Tsr.ofFloats().withShape(2, 3).andFill(1f, 4f, -2f)
            Nda<String> s = Nda.of(String.class).withShape(2, 3).andFill("a", "b", "c")

        when : 'Tensors are ND-arrays (but NdArrays are not Tensors):'
            Nda<Float> f = t // The other way around is not possible.
            //Tsr<String> c = s // The nd-array of strings is also not a tensor.
        then : 'We can confirm that all of them are ultimately just nd-arrays:'
            t instanceof Nda
            f instanceof Nda
            s instanceof Nda
    }

    def 'We can use tensors for numeric calculations (but not nd-arrays).'()
    {
        given : 'A tensor of floats and an nd-array of strings:'
            Tsr<Float>  a = Tsr.of(42f, -7f, 90f)
            Nda<String> b = Nda.of("a", "b", "c")

        when : 'We perform some numeric operations on the tensor:'
            Tsr<Float> c = a + 1f // This does not work with nd-arrays.
        then : 'The involved variables consist of the following items:'
            a.items == [42f, -7f, 90f]
            b.items == ["a", "b", "c"]
            c.items == [43f, -6f, 91f]
    }

}
