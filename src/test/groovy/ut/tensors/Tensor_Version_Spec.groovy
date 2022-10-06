package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor (Data Array) Version")
@Narrative('''

    There are two fundamental categories of operations
    which can be applied to tensors : 
    Inline operations and Non-Inline  operations! 
    
    Inline operations are often times problematic because they produce
    side effects by changing passed tensors instead of producing new ones... 
    One such bad side effect can easily occur for tensors involved in the
    autograd system, more specifically: the recorded computation graph. 
    Inline operations can break the mathematically pureness of the back-propagation
    procedure by for example changing partial derivatives... <br>
    In order to prevent said errors from occurring unnoticed tensors
    have versions which will increment when the underlying data of the tensor changes. 
    This version will be tracked by the computation graph as well in order to
    match it with the ones stored inside the tensor. 
    A mismatch would then yield an exception! 
    
    This specification is responsible for defining the behaviour of this
    version number with respect to their wrapping tensors as well as computation graph nodes.
                

''')
@Subject([Tsr])
class Tensor_Version_Spec extends Specification
{
    def setupSpec() {
        reportHeader """
                <p>
                    Here you can find out how the version number of a tensor is
                    set and tracked.
                </p>
            """
    }

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

    def 'Inline operations causes version incrementation.'(
            String code,
            boolean safe_inline,
            int version_of_c,
            int version_of_a,
            int version_of_b,
            String expected
    ) {
        given : '2 tensors a and b.'
            Neureka.get().settings().autograd().setIsPreventingInlineOperations( safe_inline )
            Tsr a = Tsr.of(4d) + Tsr.of(2d)
            Tsr b = Tsr.of(-1d) + Tsr.of(-3d).setRqsGradient(true)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect : 'Initially both tensors have a version number of 0.'
            a.getVersion() == 0
            b.getVersion() == 0

        when : 'The groovy code (performing inline operations) is being evaluated.'
            Tsr c = new GroovyShell(binding).evaluate((code))

        then : 'The resulting tensor (toString) will contain the expected sub-string.'
            c.toString().contains(expected)
            c == a

        and : 'The three tensors have the expected versions.'
            a.getVersion() == version_of_a
            b.getVersion() == version_of_b
            c.getVersion() == version_of_c

        where :
            code                     | safe_inline || version_of_c | version_of_a | version_of_b | expected
            ' a.mut.plusAssign(b) '  |   true      ||      1       |      1       |      0       | "(1):[2.0]"
            ' a.mut.minusAssign(b) ' |   true      ||      1       |      1       |      0       | "(1):[10.0]"
            ' a.mut.timesAssign(b) ' |   true      ||      1       |      1       |      0       | "(1):[-24.0]"
            ' a.mut.divAssign(b) '   |   true      ||      1       |      1       |      0       | "(1):[-1.5]"

            ' a.mut.plusAssign(b) '  |   false     ||      0       |      0       |      0       | "(1):[2.0]"
            ' a.mut.minusAssign(b) ' |   false     ||      0       |      0       |      0       | "(1):[10.0]"
            ' a.mut.timesAssign(b) ' |   false     ||      0       |      0       |      0       | "(1):[-24.0]"
            ' a.mut.divAssign(b) '   |   false     ||      0       |      0       |      0       | "(1):[-1.5]"
    }


    def 'Non-inline operations do not cause version incrementation.'(
            String code,
            boolean no_inline,
            int version_of_c,
            int version_of_a,
            int version_of_b,
            String expected
    ) {
        given :
            Neureka.get().settings().autograd().setIsPreventingInlineOperations( no_inline )
        and : 'Two tensors, one requiring gradients and the other one not.'
            Tsr a = Tsr.of(6d).setRqsGradient(true)
            Tsr b = Tsr.of(-4d)
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
            ' a % b '  |   false     ||      0       |      0       |      0       | "(1):[2.0]"

            ' a + b '  |   true      ||      0       |      0       |      0       | "(1):[2.0]"
            ' a - b '  |   true      ||      0       |      0       |      0       | "(1):[10.0]"
            ' a * b '  |   true      ||      0       |      0       |      0       | "(1):[-24.0]"
            ' a / b '  |   true      ||      0       |      0       |      0       | "(1):[-1.5]"
            ' a % b '  |   true      ||      0       |      0       |      0       | "(1):[2.0]"
    }



    def 'Inline operations cause illegal state exceptions.'(
            String code,
            String message
    ) {
        given :
            Neureka.get().settings().autograd().setIsPreventingInlineOperations( true )
            Tsr a = Tsr.of(4d) + Tsr.of(2d).setRqsGradient(true)
            Tsr b = Tsr.of(-4d)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect :
            a.getVersion() == 0
            b.getVersion() == 0

        when : 'The groovy code is being evaluated.'
            Tsr<Double> c = new GroovyShell( binding ).evaluate( code )

        then : 'An illegal state exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == message

        and : 'The variable "c" is null!'
            c == null

        where :
            code                  ||  message
            'a.mut.plusAssign(b) '|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.mut.minusAssign(b)'|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.mut.timesAssign(b)'|| "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
            'a.mut.divAssign(b) ' || "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\nThe following OperationType caused an internal version mismatch: 'left_inline'"
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() })
    def 'Storing a tensor on a device should not change the version of a tensor (Even though its data changed technically).'()
    {
        given :
            var t = Tsr.ofFloats().withShape(5, 2).andSeed(42)
        expect :
            t.version == 0

        when :
            t.to('GPU')
        then :
            t.isOutsourced()
            t.version == 0

        when :
            t[0] = 1f
            t[new int[]{1, 0}] = 6f
            t[0, 1] = 42f
            t.set(0, 3f)
            t.setItemAt(3, 4f)
            t.set(new int[]{1, 1}, -1f)
            t.mut.setDataAt(3, 5f)
        then :
            t.version == 7
    }

}
