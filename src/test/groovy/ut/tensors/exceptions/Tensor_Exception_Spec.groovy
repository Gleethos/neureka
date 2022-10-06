package ut.tensors.exceptions

import groovy.transform.CompileDynamic
import neureka.Neureka
import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Title

import java.util.function.Consumer

@Title('Tensors Exception Behavior')
@Narrative('''

    This specification covers the behavior of the $Tsr class in
    exceptional scenarios which are contrary to its intended use.
    The purpose of this is to assert that the $Tsr class will provide
    useful feedback to a user to explain that a misuse of its API
    occurred so that the user can correct this misuse.

''')
@CompileDynamic
class Tensor_Exception_Spec extends Specification
{

    @Shared PrintStream oldStream

    def setupSpec() {
        reportHeader """
                <h2> Tensors Exception Behavior </h2>
                <br> 
                <p>
                    This specification covers the behavior of tensors
                    in exceptional situations.           
                </p>
            """
    }

    def setup() {
        Neureka.get().reset()
        oldStream = System.err
        System.err = Mock(PrintStream)
    }

    def cleanup() {
        System.err = oldStream
    }

    def "Trying to inject an empty tensor into another causes fitting exception."()
    {
        given : 'A new tensor instance used for exception testing.'
            Tsr<Integer> t = Tsr.of([6, 6], -1)

        when : 'We try to inject an empty tensor whose size does of course not match...'
            t.mut[[1..3], [1..3]] = Tsr.newInstance() as Tsr<Integer>

        then : 'The expected message is being thrown, which tells us that '
            def exception = thrown(IllegalArgumentException)
            assert exception.message.contains("Provided tensor is empty! Empty tensors cannot be injected.")
    }

    def 'Passing null to various methods of the tensor API will throw exceptions.'(
            Class<Exception> type, Consumer<Tsr<Integer>> errorCode
    ) {
        given :
            Tsr<Integer> t = Tsr.of(1, 2, 3)

        when :
            errorCode(t)

        then :
            var exception = thrown(type)
        and :
            exception.message != "" && exception.message.length() > 13

        where :
            type                        | errorCode
            IllegalArgumentException    | { Tsr x -> x.times((Tsr)null) }
            IllegalArgumentException    | { Tsr x -> x.div((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.plus((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mod((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mut.timesAssign((Tsr)null) }
            IllegalArgumentException    | { Tsr x -> x.mut.divAssign((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mut.plusAssign((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mut.modAssign((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mut.minusAssign((Tsr)null)  }
            IllegalArgumentException    | { Tsr x -> x.mut.label((String[][])null) }
            IllegalArgumentException    | { Tsr x -> x.mut.label("hi", (String[][])null) }
            IllegalArgumentException    | { Tsr x -> x.mut.label(null, (String[][])null) }
            IllegalArgumentException    | { Tsr x -> x.mut.label("hi", (Map)null) }
            IllegalArgumentException    | { Tsr x -> x.mut.label(null, (Map)null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels("hi", (String[][])null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels(null, (String[][])null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels(null, (String[])null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels((String[])null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels("hi", (Map)null) }
            IllegalArgumentException    | { Tsr x -> x.withLabels(null, (Map)null) }
    }

    def 'Passing an invalid object into Tsr constructor causes descriptive exception.'()
    {
        when : 'A tensor is being instantiated with a nonsensical parameter.'
            Tsr.ofRandom(Scanner.class, 2, 4)
        then : 'An exception is being thrown which tells us about it.'
            def exception = thrown(IllegalArgumentException)
            exception.message.contains(
                    "Could not create a random tensor for type 'class java.util.Scanner'!"
            )
    }


    def 'Passing an invalid key object into the "getAt" method causes a descriptive exception.'()
    {
        given : 'A new test tensor is being instantiated.'
            Tsr t = Tsr.of( [2, 3], -1..6 )

        when : 'A nonsensical object is being passed to the tensor.'
            t[ [null] ]

        then : 'An exception is being thrown which tells us about it.'
            def exception = thrown(IllegalArgumentException)
            exception.message.contains("List of indices/ranges may not contain entries which are null!")
    }


    def 'Out of dimension bound causes descriptive exception!'()
    {
        when : 'Some more complex slicing is being performed...'
            Tsr t = Tsr.of( [3, 3, 3, 3], 0 )
            t.mut[1..2, 1..3, 1..1, 0..2] = Tsr.of( [2, 3, 1, 3], -4..2 )

        then : 'The slice range 1..3 causes and exception!'
            def exception = thrown(IllegalArgumentException)
            exception.message == "java.lang.IllegalArgumentException: " +
                    "Cannot create slice because ranges are out of the bounds of the targeted tensor.\n" +
                    "At index '1' : offset '1' + shape '3' = '4',\n" +
                    "which is larger than the target shape '3' at the same index!"

        and : 'The logger logs the exception message!'
            1 * System.err.println({it.endsWith(
                        "Cannot create slice because ranges are out of the bounds of the targeted tensor.\n" +
                        "At index '1' : offset '1' + shape '3' = '4',\n" +
                        "which is larger than the target shape '3' at the same index!")
                    })
    }

    def 'Building a tensor with 0 shape arguments throws an exception.'() {

        when :
            var t = Tsr.ofInts().withShape().all(0)

        then :
            thrown(IllegalArgumentException)
    }

    def 'Casting a tensor as something unusual will cuas an exception to be thrown.'()
    {
        given : 'We have a regular tensor of 2 bytes!'
            var t = Tsr.ofBytes().withShape(2).andFill(-1, 2)

        when : 'We try to convert the tensor to an instance of type "Random"...'
            t as Random

        then : 'This will obviously least to an exception being thrown.'
            thrown(IllegalArgumentException)
    }

    def 'Building a tensor with "null" as shape argument throws an exception.'()
    {
        when :
            Tsr.ofInts().withShape((List)null).all(0)
        then :
            thrown(IllegalArgumentException)

        when :
            Tsr.ofInts().withShape((int[])null).all(0)
        then :
            thrown(IllegalArgumentException)
    }

}
