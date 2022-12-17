package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Expression based Tensor Instantiation")
@Narrative('''
 
    This specification defines how a tensor can be instantiated
    using string expressions, which define operations to be executed.
    This form of tensor instantiation is very useful to avoid boilerplate code.
    
''')
@Subject([Tsr])
class Expression_Based_Tensor_Instantiation_Spec extends Specification
{
    def setup() {
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

    def 'A tensor can be created from a function as expression.'()
    {
        reportInfo """
            The `Tsr.of` method can be used to instantiate a tensor
            using a string expression which defines a function 
            followed by an arbitrary number of tensor arguments
            which are used as input for the function.
        """

        given : 'A simple scalar tensor containing the number "4".'
            Tsr<Double> x = Tsr.of(4d)

        when : 'We instantiate a tensor using a function expression and the scalar tensor as argument...'
            Tsr<Double> y = Tsr.of("tanh(I[0])", x)

        then : 'The resulting tensor should be the result of the "tanh" function applied to the scalar tensor.'
            y.toString() == "(1):[0.99932]"
        and : 'We also expect the following lines to be true:'
            y.isBranch()
            !y.isLeave()
            y.belongsToGraph()
            x.belongsToGraph() // <- This is true because the tensor x is used as argument for the function.
    }

    def 'We can instantiate tensors from various simple string expressions.'()
    {
        given : 'Three scalar tensors.'
            Tsr<Double> a = Tsr.of(3d)
            Tsr<Double> b = Tsr.of(2d)
            Tsr<Double> c = Tsr.of(-1d)

        when : var t = Tsr.of("1+", a, "*", b)
        then : t.toString().contains("7.0")
        when : t = Tsr.of("1", "+", a, "*", b)
        then : t.toString().contains("7.0")
        when : t = Tsr.of("(","1+", a,")", "*", b)
        then : t.toString().contains("8.0")
        when : t = Tsr.of("(","1", "+", a,")", "*", b)
        then : t.toString().contains("8.0")
        when : t = Tsr.of("(", c, "*3)+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")
        when : t = Tsr.of("(", c, "*","3)+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")
        when : t = Tsr.of("(", c, "*","3", ")+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")
    }

}
