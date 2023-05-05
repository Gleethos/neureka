package ut.ndim

import neureka.Neureka
import neureka.Tsr
import neureka.ndim.config.types.simple.Simple1DConfiguration
import neureka.ndim.config.types.simple.Simple0DConfiguration
import neureka.ndim.config.types.sliced.Sliced0DConfiguration
import neureka.ndim.config.types.sliced.Sliced1DConfiguration
import neureka.view.NDPrintSettings
import spock.lang.Specification
import spock.lang.Title
import spock.lang.Narrative

@Title("What it means to be N-Dimensional")
@Narrative('''

    This specification covers how implementations
    of the `NDConfiguration` interface manage to define
    what it means to be a n-dimensional tensor/nd-array.

''')
class Tensor_NDConfiguration_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
                For certain situations the "Tsr" class should use the correct 
                implementations of said interface as configuration for internal index mapping...

        """
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'NDConfiguration instances of tensors have expected state.'()
    {
        given: 'Two scalar tensors "a" and "b" storing values "1" and "2".'
            Tsr a = Tsr.of(1d)
            Tsr b = Tsr.of(2d)

        expect: 'Tensor "a" contains an instance of the "SimpleScalarConfiguration".'
            a.NDConf instanceof Simple0DConfiguration
        and : 'Both tensors "a" and "b" share the same (cached) "NDConfiguration" instance because they are both scalars.'
            a.NDConf == b.NDConf
        and : 'This ND-Configuration has the expected state.'
            a.NDConf.shape(0) == 1
            a.NDConf.strides(0) == 1
            a.NDConf.indicesMap(0) == 1
            a.NDConf.offset(0) == 0
            a.NDConf.spread(0) == 1
    }


    def 'NDConfiguration instances of tensors have expected state and behaviour.'()
    {
        given: 'Three vector tensors containing different numeric values.'
            Tsr<Object> x = Tsr.of([1.0, 2.0, 3.1])
            Tsr<Object> y = Tsr.of([3, 4.5, 2])
            Tsr<Object> z = Tsr.of([1.4, 2, 4])

        expect : 'All of them possess "SimpleD1Configuration" NDConfiguration implementations.'
            x.NDConf instanceof Simple1DConfiguration
            y.NDConf instanceof Simple1DConfiguration
            z.NDConf instanceof Simple1DConfiguration
        and : 'They all share the same (cached) SimpleD1Configuration instance because they do not require otherwise.'
            x.NDConf == y.NDConf
            y.NDConf == z.NDConf
        and : 'The configuration behaves as expected.'
            x.NDConf.shape(0) == 3
            x.NDConf.strides(0) == 1
            x.NDConf.indicesMap(0) == 1
            x.NDConf.offset(0) == 0
            x.NDConf.spread(0) == 1
        and : 'Also, scalar slices have the expected configs'
            x[2].NDConf instanceof Sliced0DConfiguration
            y[1.1].NDConf instanceof Sliced0DConfiguration
            y[1.1].NDConf != x[2].NDConf
            y[1.1].NDConf == z[1].NDConf

        when : 'We try to extract a slice by using a BigDecimal instance...'
            x = x[new BigDecimal(2)]
        then : 'This also produces a valid slice with the expected properties :'
            x.NDConf.shape(0) == 1
            x.NDConf.strides(0) == 1
            x.NDConf.indicesMap(0) == 1
            x.NDConf.offset(0) == 2
            x.NDConf.spread(0) == 1

        when : 'We try using a Range instance to extract a slice...'
            y = y[1..2]
        then : 'This produces the expected slice.'
            y.toString().contains("(2):[4.5, 2.0]")
        and : 'The NDConfiguration of this slice has the expected state.'
            y.NDConf instanceof Sliced1DConfiguration
            y.NDConf.shape(0) == 2
            y.NDConf.strides(0) == 1
            y.NDConf.indicesMap(0) == 1
            y.NDConf.offset(0) == 1
            y.NDConf.spread(0) == 1

    }


}
