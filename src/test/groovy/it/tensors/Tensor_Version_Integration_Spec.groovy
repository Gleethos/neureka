package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.view.TsrStringSettings
import spock.lang.Specification

class Tensor_Version_Integration_Spec extends Specification
{
    def setupSpec() {
        reportHeader """
                <h2> Tensor Version Behavior </h2>
                <br> 
                <p>
                    There are two fundamental categories of operations
                    which can be applied to tensors : <br>
                    Inline operations and Non-Inline  operations! <br>
                    <br>
                    Inline operations are often times problematic because they produce
                    side effects by changing passed tensors instead of producing new ones... <br>   
                    One such bad side effect can easily occur for tensors involved in the
                    autograd system, more specifically: the recorded computation graph. <br>
                    Inline operations can break the mathematically pureness of the back-propagation
                    procedure by for example changing partial derivatives... <br>
                    In order to prevent said errors to occur unnoticed tensors
                    have versions which will increment when the underlying data of the tensor changes. <br>
                    This version will be tracked by the computation graph as well in order to
                    match it with the ones stored inside the tensor. <br>
                    A mismatch would then yield an exception! <br>
                    <br>
                    This specification is responsible for defining the behaviour of this
                    version number with respect to their wrapping tensors as well as computation graph nodes.
                
                </p>
            """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    def 'Non-inline operations causes version incrementation.'(
            String code,
            boolean no_inline,
            int version_of_c,
            int version_of_a,
            int version_of_b,
            String expected
    ) {
        given : 'Two tensors, one requiring gradients and the other one not.'
            Tsr a = Tsr.of(6).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
        and : 'A binding for both tensors as preparation for calling the Groovy shell.'
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect : 'The versions of both tensors are 0 initially.'
            a.getVersion() == 0
            b.getVersion() == 0

        when : 'The Groovy code is being evaluated inside the Groovy shell.'
            Tsr c = new GroovyShell(binding).evaluate((code))

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().contains(expected)
            c != a

        and : 'The three tensors have the expected versions.'
            a.getVersion() == version_of_a
            b.getVersion() == version_of_b
            c.getVersion() == version_of_c

        where : 'The following arguments are being used:'
            code       |  no_inline  || version_of_c | version_of_a | version_of_b | expected
            ' a + b '  |   false     ||      0       |      0       |      0       | "(1):[2.0]"
            ' a - b '  |   false     ||      0       |      0       |      0       | "(1):[10.0]"
            ' a * b '  |   false     ||      0       |      0       |      0       | "(1):[-24.0]"
            ' a / b '  |   false     ||      0       |      0       |      0       | "(1):[-1.5]"

            ' a + b '  |   true      ||      0       |      0       |      0       | "(1):[2.0]"
            ' a - b '  |   true      ||      0       |      0       |      0       | "(1):[10.0]"
            ' a * b '  |   true      ||      0       |      0       |      0       | "(1):[-24.0]"
            ' a / b '  |   true      ||      0       |      0       |      0       | "(1):[-1.5]"
    }


    def 'Inline operations causes version incrementation.'(
            String code,
            boolean save_inline,
            int version_of_c,
            int version_of_a,
            int version_of_b,
            String expected
    ) {
        given :
            Neureka.get().settings().autograd().setIsPreventingInlineOperations( save_inline )
            Tsr a = Tsr.of(4) + Tsr.of(2)
            Tsr b = Tsr.of(-1) + Tsr.of(-3).setRqsGradient(true)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect :
            a.getVersion() == 0
            b.getVersion() == 0

        when : 'The groovy code is being evaluated.'
            Tsr c = new GroovyShell(binding).evaluate((code))

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().contains(expected)
            c == a

        and : 'The three tensors have the expected versions.'
            a.getVersion() == version_of_a
            b.getVersion() == version_of_b
            c.getVersion() == version_of_c

        where :
            code                  | save_inline || version_of_c | version_of_a | version_of_b | expected
            ' a.plusAssign(b) '   |   true      ||      1       |      1       |      0       | "(1):[2.0]"
            ' a.minusAssign(b) '  |   true      ||      1       |      1       |      0       | "(1):[10.0]"
            ' a.timesAssign(b) '  |   true      ||      1       |      1       |      0       | "(1):[-24.0]"
            ' a.divAssign(b) '    |   true      ||      1       |      1       |      0       | "(1):[-1.5]"

            ' a.plusAssign(b) '   |   false     ||      0       |      0       |      0       | "(1):[2.0]"
            ' a.minusAssign(b) '  |   false     ||      0       |      0       |      0       | "(1):[10.0]"
            ' a.timesAssign(b) '  |   false     ||      0       |      0       |      0       | "(1):[-24.0]"
            ' a.divAssign(b) '    |   false     ||      0       |      0       |      0       | "(1):[-1.5]"
    }


    def 'Inline operations cause illegal state exceptions.'(
            String code,
            String message
    ) {
        given :
            Neureka.get().settings().autograd().setIsPreventingInlineOperations( true )
            Tsr a = Tsr.of(4) + Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect :
            a.getVersion() == 0
            b.getVersion() == 0

        when : 'The groovy code is being evaluated.'
            Tsr c = new GroovyShell( binding ).evaluate( code )

        then : 'An illegal state exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == message

        and : 'The variable "c" is null!'
            c == null

        where :
            code               ||  message
            'a.plusAssign(b) '|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.minusAssign(b)'|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.timesAssign(b)'|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.divAssign(b) ' || "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
    }


}
