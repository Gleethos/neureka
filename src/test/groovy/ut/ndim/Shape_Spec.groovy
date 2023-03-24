package ut.ndim

import neureka.Nda
import neureka.Shape
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("The Shape Tuple")
@Narrative('''
    The `Shape` of an nd-array/tensor is in essence merely an immutable tuple of integers
    which define the size of each dimension of the tensor.
    So if you think of an nd-array as a grid of numbers, then the shape of the
    tensor is the size of the grid in each dimension.
    
    This specifications shows you how to create a shape and how to use it.
''')
@Subject([Shape])
class Shape_Spec extends Specification
{
    def 'A shape can be created from a list of integers.'() {
        given : 'We use the "of" method to create a shape from a list of integers.'
            var shape = Shape.of( 2, 3, 4 )
        expect : 'It has the expected size and values!'
            shape.size() == 3
            shape.get(0) == 2
            shape.get(1) == 3
            shape.get(2) == 4
    }

    def 'A shape can be created from a stream of ints.'() {
        given : 'We use the "of" method to create a shape from a stream of integers.'
            var shape = Shape.of( [2, 3, 4].stream() )
        expect : 'It has the expected size and values!'
            shape.size() == 3
            shape.get(0) == 2
            shape.get(1) == 3
            shape.get(2) == 4
    }

    def 'A shape can be created from an iterable.'()
    {
        given : 'We use the "of" method to create a shape from a stream of integers.'
            var shape = Shape.of( Nda.of( 2, 3, 4 ) )
        expect : 'It has the expected size and values!'
            shape.size() == 3
            shape.get(0) == 2
            shape.get(1) == 3
            shape.get(2) == 4
    }

    def 'A shape can be mapped to a new shape.'()
    {
        reportInfo """
            Note that as a tuple, the shape is immutable so you cannot change its values.
            But as a monad, the shape can be mapped to a new shape
            using the "map" method. :)
        """
        given : 'We use the "of" method to create a shape from a list of integers.'
            var shape = Shape.of( 2, 3, 4 )
        when : 'We multiply each value of the shape by 2 into a new shape.'
            var newShape = shape.map( { it * 2 } )
        then : 'The new shape has the expected size and values!'
            newShape.size() == 3
            newShape.get(0) == 4
            newShape.get(1) == 6
            newShape.get(2) == 8
    }

    def 'A shape can be sliced.'()
    {
        reportInfo """
            This is similar as the "subList" method of the java.util.List interface.
            It returns a new shape which is a slice of the original shape
            starting at the given index and ending at the given index.
        """
        given : 'We use the "of" method to create a shape from a list of integers.'
            var shape = Shape.of( 2, 3, 4 )
        when : 'We slice the shape from index 1 to index 2.'
            var newShape = shape.slice( 1, 2 )
        then : 'The new shape has the expected size and values!'
            newShape.size() == 1
            newShape.get(0) == 3
    }

}
