package ut.ndim


import neureka.ndim.config.NDConfiguration
import neureka.ndim.config.types.permuted.Permuted1DConfiguration
import neureka.ndim.config.types.permuted.Permuted2DConfiguration
import neureka.ndim.config.types.permuted.Permuted3DConfiguration
import neureka.ndim.config.types.permuted.PermutedNDConfiguration
import neureka.ndim.config.types.simple.Simple1DConfiguration
import neureka.ndim.config.types.simple.Simple2DConfiguration
import neureka.ndim.config.types.simple.Simple3DConfiguration
import neureka.ndim.config.types.simple.SimpleNDConfiguration
import neureka.ndim.config.types.sliced.Sliced1DConfiguration
import neureka.ndim.config.types.sliced.Sliced2DConfiguration
import neureka.ndim.config.types.sliced.Sliced3DConfiguration
import neureka.ndim.config.types.sliced.SlicedNDConfiguration
import neureka.ndim.iterator.NDIterator
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Making Arrays N-Dimensional")
@Narrative('''

    Under the hood Neureka implements powerful indexing 
    abstractions through the `NDConfiguration` interface and its various implementations.
    This allows for the creation of tensors/nd-arrays with arbitrary dimensions, 
    the ability to slice them into smaller tensors/nd-arrays with the same underlying data,
    and finally the ability to permute their axes (like transposing them for example).
    
    This specification however only focuses on the behaviour of the `NDConfiguration` interface
    which translates various types of indices.

''')
@Subject([NDConfiguration])
class NDConfiguration_Spec extends Specification
{
    def 'Various NDConfigurations behaviour exactly as their general purpose implementation.'(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset,
            Class<?> expected
    ) {
        given : 'A general purpose NDConfiguration implementation as well as a specialized one (provided by the factory method).'
            var ndc1 = SlicedNDConfiguration.construct(shape, translation, indicesMap, spread, offset)
            var ndc2 = NDConfiguration.of(shape, translation, indicesMap, spread, offset)
        and : '2 corresponding iterators:'
            var i1 = NDIterator.of(ndc1, NDIterator.NonVirtual.FALSE)
            var i2 = NDIterator.of(ndc2, NDIterator.NonVirtual.FALSE)

        expect:
            ndc2.getClass()    == expected
            ndc1.rank()        == ndc2.rank()
            ndc1.size()        == ndc2.size()
            ndc1.shape()       == ndc2.shape()
            ndc1.translation() == ndc2.translation()
            ndc1.indicesMap()  == ndc2.indicesMap()
            ndc1.spread()      == ndc2.spread()
            ndc1.offset()      == ndc2.offset()

        and :
            (0..ndc1.size()-1).collect({
                ndc1.indicesOfIndex(it) == ndc2.indicesOfIndex(it)
            })
            .every()
        and :
            (0..ndc1.size()-1).collect({
                ndc1.indexOfIndices(ndc1.indicesOfIndex(it)) == ndc2.indexOfIndices(ndc2.indicesOfIndex(it))
            })
            .every()
        and :
            (0..ndc1.size()-1).collect({
                ndc1.indexOfIndex(it) == ndc2.indexOfIndex(it)
            })
            .every()
        and :
            (0..ndc1.size()-1).collect({
                boolean matches = i1.get() == i2.get()
                if ( it < ndc1.size()-1 ) { i1.increment(); i2.increment() }
                return matches
            })
            .every()
        and :
           (0..ndc1.size()-1).collect({
               boolean matches = i1.get() == i2.get()
               i1.decrement(); i2.decrement()
               return matches
           })
           .every()

        where :
            shape     | translation    | indicesMap     | spread     | offset    || expected
            [2,3,8,4] | [96, 32, 4, 1] | [96, 32, 4, 1] | [1,1,1,1]  | [0,0,0,0] || SimpleNDConfiguration
            [2,3,8,4] | [96, 200, 8, 1]| [96, 32, 4, 1] | [1,1,1,1]  | [0,0,0,0] || PermutedNDConfiguration
            [2,3,8,4] | [96, 32, 4, 1] | [96, 92, 4, 1] | [1,4,1,1]  | [0,0,0,0] || SlicedNDConfiguration

            [2,3,8]   | [24,8,1]       | [24,8,1]       | [1, 1, 1]  | [0,0,0]   || Simple3DConfiguration
            [2,3,8]   | [8,24,7]       | [1,2,3]        | [1, 1, 1]  | [0,0,0]   || Permuted3DConfiguration
            [2,3,8]   | [8,24,7]       | [1,2,3]        | [1, 7, 1]  | [0,0,0]   || Sliced3DConfiguration

            [2,3]     | [3,1]          | [3,1]          | [1, 1]     | [0,0]     || Simple2DConfiguration
            [2,3]     | [1,2]          | [1,2]          | [1, 1]     | [0,0]     || Permuted2DConfiguration
            [2,3]     | [1,2]          | [3,1]          | [1, 1]     | [0,0]     || Permuted2DConfiguration
            [2,3]     | [81,42]        | [3,99]         | [1, 1]     | [0,0]     || Permuted2DConfiguration
            [2,3]     | [1,2]          | [2,1]          | [7, 2]     | [1,8]     || Sliced2DConfiguration
            [2,3]     | [3,1]          | [3,1]          | [1, 1]     | [6,0]     || Sliced2DConfiguration
            [2,3]     | [3,1]          | [3,1]          | [1, 2]     | [0,0]     || Sliced2DConfiguration

            [3]       | [1]            | [1]            | [1]        | [0]       || Simple1DConfiguration
            [42]      | [1]            | [1]            | [1]        | [0]       || Simple1DConfiguration
            [3]       | [1]            | [2]            | [1]        | [0]       || Permuted1DConfiguration
            [30]      | [8]            | [2]            | [1]        | [0]       || Permuted1DConfiguration
            [2]       | [1]            | [1]            | [1]        | [5]       || Sliced1DConfiguration
    }


}


