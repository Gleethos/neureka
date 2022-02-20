package ut.ndim

import neureka.ndim.config.AbstractNDC
import neureka.ndim.config.NDConfiguration
import neureka.ndim.config.types.simple.Simple2DConfiguration
import neureka.ndim.config.types.sliced.Sliced2DConfiguration
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

        when :
            (generalized.size()*0.75).times {
                i1.increment(); i2.increment()
            }

        then :
            i1.get() == i2.get()

        where :
            shape | translation | indicesMap | spread | offset || expected
            [2,3] | [1,2]       | [1,2]      | [1, 1] | [0,0]  || Sliced2DConfiguration
            [2,3] | [1,2]       | [2,1]      | [7, 2] | [1,8]  || Sliced2DConfiguration
    }



}


