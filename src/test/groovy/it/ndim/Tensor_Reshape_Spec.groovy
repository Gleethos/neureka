package it.ndim

import neureka.Tsr
import neureka.ndim.config.NDConfiguration
import spock.lang.Specification

class Tensor_Reshape_Spec extends Specification {

    def 'When matrices are transpose, they will change their layout type.'() {

        given :
            Tsr t = Tsr.ofFloats().withShape(3, 4).andSeed(42)

        expect :
            t.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR

        when :
            t = t.T

        then :
            t.NDConf.layout == NDConfiguration.Layout.COLUMN_MAJOR

    }

}
