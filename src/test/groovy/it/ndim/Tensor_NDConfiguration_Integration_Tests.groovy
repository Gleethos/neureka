package it.ndim

import neureka.Neureka
import neureka.Tsr
import neureka.ndim.config.types.complex.ComplexD1Configuration
import neureka.ndim.config.types.complex.ComplexScalarConfiguration
import neureka.ndim.config.types.simple.SimpleD1Configuration
import neureka.ndim.config.types.simple.SimpleScalarConfiguration
import spock.lang.Specification

class Tensor_NDConfiguration_Integration_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> Tensor NDConfiguration Integration Tests </h2>
            <p>
                Specified below are strict integration tests for tensors and
                their behaviour with regards to the usage of implementations of the
                NDConfiguration interface. <br>
                For certain situations the "Tsr" class should use the correct 
                implementations of said interface as configuration for internal index mapping...
            </p>
        """
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'NDConfiguration instances of tensors have expected state.'()
    {
        given: 'Two scalar tensors "a" and "b" storing values "1" and "2".'
            Tsr a = Tsr.of(1)
            Tsr b = Tsr.of(2)

        expect: 'Tensor "a" contains an instance of the "SimpleScalarConfiguration".'
            a.NDConf instanceof SimpleScalarConfiguration
        and : 'Both tensors "a" and "b" share the same (cached) "NDConfiguration" instance because they are both scalars.'
            a.NDConf == b.NDConf
        and : 'This ND-Configuration behaves as expected.'
            a.NDConf.shape(0) == 1
            a.NDConf.translation(0) == 1
            a.NDConf.indicesMap(0) == 1
            a.NDConf.offset(0) == 0
            a.NDConf.spread(0) == 1
    }


    def 'NDConfiguration instances of tensors have expected state and behaviour.'()
    {
        given: 'Three vector tensors containing different numeric values.'
            Tsr x = Tsr.of([1.0, 2.0, 3.1])
            Tsr y = Tsr.of([3, 4.5, 2])
            Tsr z = Tsr.of([1.4, 2, 4])

        expect : 'All of them possess "SimpleD1Configuration" NDConfiguration implementations.'
            x.NDConf instanceof SimpleD1Configuration
            y.NDConf instanceof SimpleD1Configuration
            z.NDConf instanceof SimpleD1Configuration
        and : 'They all share the same (cached) SimpleD1Configuration instance because they do not require otherwise.'
            x.NDConf == y.NDConf
            y.NDConf == z.NDConf
        and : 'The configuration behaves as expected.'
            x.NDConf.shape(0) == 3
            x.NDConf.translation(0) == 1
            x.NDConf.indicesMap(0) == 1
            x.NDConf.offset(0) == 0
            x.NDConf.spread(0) == 1
            x[2].NDConf instanceof ComplexScalarConfiguration
            y[1.1].NDConf instanceof ComplexScalarConfiguration
            y[1.1].NDConf != x[2].NDConf
            y[1.1].NDConf == z[1].NDConf

        when : 'We try to extract a slice by using a BigDecimal instance...'
            x = x[new BigDecimal(2)]
        then : 'This also produces a valid slice with the expected properties :'
            x.NDConf.shape(0) == 1
            x.NDConf.translation(0) == 1
            x.NDConf.indicesMap(0) == 1
            x.NDConf.offset(0) == 2
            x.NDConf.spread(0) == 1

        when : 'We try using a Range instance to extract a slice...'
            y = y[1..2]
        then : 'This produces the expected slice.'
            y.toString().contains("(2):[4.5, 2.0]")
            y.NDConf instanceof ComplexD1Configuration
            y.NDConf.shape(0) == 2
            y.NDConf.translation(0) == 1
            y.NDConf.indicesMap(0) == 1
            y.NDConf.offset(0) == 1
            y.NDConf.spread(0) == 1

    }


}
