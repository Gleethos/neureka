package ut.ndim

import neureka.ndim.config.AbstractNDC
import neureka.ndim.config.NDConfiguration
import neureka.ndim.config.types.simple.Simple1DConfiguration
import neureka.ndim.config.types.simple.Simple2DConfiguration
import neureka.ndim.config.types.simple.Simple3DConfiguration
import neureka.ndim.config.types.simple.SimpleNDConfiguration
import neureka.ndim.config.types.sliced.Sliced1DConfiguration
import neureka.ndim.config.types.sliced.Sliced2DConfiguration
import neureka.ndim.config.types.sliced.Sliced3DConfiguration
import neureka.ndim.config.types.sliced.SlicedNDConfiguration
import neureka.ndim.iterators.NDIterator
import spock.lang.Specification

class NDConfiguration_Spec extends Specification {


    def 'Various NDConfigurations behaviour exactly as their general purpose implementation.'(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset,
            Class<?> expected
    ) {

        given :
            var generalized = SlicedNDConfiguration.construct(shape, translation, indicesMap, spread, offset)
            var specialized = AbstractNDC.construct(shape, translation, indicesMap, spread, offset, NDConfiguration.Layout.ROW_MAJOR)
            var i1 = NDIterator.of(generalized, NDIterator.NonVirtual.FALSE)
            var i2 = NDIterator.of(specialized, NDIterator.NonVirtual.FALSE)

        expect:
            specialized.getClass()    == expected
            generalized.rank()        == specialized.rank()
            generalized.size()        == specialized.size()
            generalized.shape()       == specialized.shape()
            generalized.translation() == specialized.translation()
            generalized.indicesMap()  == specialized.indicesMap()
            generalized.spread()      == specialized.spread()
            generalized.offset()      == specialized.offset()

        and :
            (0..generalized.size()-1).collect({
                var matches = i1.get() == i2.get()
                if ( it < generalized.size()-1 ) { i1.increment(); i2.increment() }
                return matches
            })
            .every()
        and :
           (0..generalized.size()-1).collect({
               var matches = i1.get() == i2.get()
               i1.decrement(); i2.decrement()
               print i1.get()
               println i2.get()
               return matches
           })
           .every()

        where :
            shape     | translation    | indicesMap     | spread     | offset    || expected
            [2,3,8,4] | [96, 32, 4, 1] | [96, 32, 4, 1] | [1,1,1,1]  | [0,0,0,0] || SimpleNDConfiguration
            [2,3,8,4] | [96, 32, 4, 1] | [96, 32, 4, 1] | [1,4,1,1]  | [0,0,0,0] || SlicedNDConfiguration

            [2,3,8]   | [24,8,1]       | [24,8,1]       | [1, 1, 1]  | [0,0,0]   || Simple3DConfiguration
            [2,3,8]   | [8,24,7]       | [1,2,3]        | [1, 1, 1]  | [0,0,0]   || Sliced3DConfiguration

            [2,3]     | [3,1]          | [3,1]          | [1, 1]     | [0,0]     || Simple2DConfiguration
            [2,3]     | [1,2]          | [1,2]          | [1, 1]     | [0,0]     || Sliced2DConfiguration
            [2,3]     | [1,2]          | [2,1]          | [7, 2]     | [1,8]     || Sliced2DConfiguration
            [2,3]     | [3,1]          | [3,1]          | [1, 1]     | [6,0]     || Sliced2DConfiguration
            [2,3]     | [3,1]          | [3,1]          | [1, 2]     | [0,0]     || Sliced2DConfiguration

            [3]       | [1]            | [1]            | [1]        | [0]       || Simple1DConfiguration
            [42]      | [1]            | [1]            | [1]        | [0]       || Simple1DConfiguration
            [3]       | [1]            | [2]            | [1]        | [0]       || Sliced1DConfiguration
            [2]       | [1]            | [1]            | [1]        | [5]       || Sliced1DConfiguration
    }



}


