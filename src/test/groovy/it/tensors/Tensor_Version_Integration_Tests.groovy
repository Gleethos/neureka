package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.utility.TsrAsString
import spock.lang.Specification

class Tensor_Version_Integration_Tests extends Specification
{


    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = "dgc"
    }

    def 'Non-inline operations causes version incrementation.'(
            String code,
            boolean no_inline,
            int version_of_c,
            int version_of_a,
            int version_of_b,
            String expected
    ) {
        given :
            Tsr a = new Tsr(6).setRqsGradient(true)
            Tsr b = new Tsr(-4)
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
            c != a

        and : 'The three tensors have the expected versions.'
            a.getVersion() == version_of_a
            b.getVersion() == version_of_b
            c.getVersion() == version_of_c

        where :
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
            Neureka.instance().settings().autograd().setIsPreventingInlineOperations( save_inline )
            Tsr a = new Tsr(4) + new Tsr(2)
            Tsr b = new Tsr(-1) + new Tsr(-3).setRqsGradient(true)
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
            Neureka.instance().settings().autograd().setIsPreventingInlineOperations( true )
            Tsr a = new Tsr(4) + new Tsr(2).setRqsGradient(true)
            Tsr b = new Tsr(-4)
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
