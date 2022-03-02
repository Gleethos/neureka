import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("An Introduction to writing Spock Specifications")
@Narrative('''

    Hello and welcome to the example / template specification of this project.
    This is a simple introduction as to how to get started writing Spock specifications.
    
    Spock works on top of Groovy which is in essence a syntactic super-set of Java.
    That means that one can write Java code in Groovy, and 99% of the time it will 
    work the exact same way.
    
''')
class Example_Spec extends Specification {

    /**
     *  This is the first example of a test in Spock & Groovy.
     *  However this is not best practice!
     *  Take a look at the next method...
     */
    def iAmNotSoReadable() {
        expect : 42 == 42
    }

    /**
     *  What you are seeing might be weired at first but fear not!
     *  In Groovy one can use Strings to define method names, Spock uses
     *  this fact to allow us to define readable names for unit tests!
     */
    def 'I am readable and also best practice!'() {
        expect : 42 == 42
    }

    /**
     *  Spock is a BDD driven framework.
     *  Therefore test classes are called specifications and individual tests are called "features"!
     *  But let's get to the point:
     *  Where is the assertion in the test below?!?
     *  Well the answer is simply that the "expect" block automatically
     *  tells Spock that the result of the following code block ought to
     *  be asserted.
     */
    def 'Call me feature not unit test!'() {
        expect : 2 == 2
    }

    /**
     *  But the "expect" is not the only Spock syntax!
     *  As expected from a BDD framework there is also a "given", "when" and "then".
     *  The "expect" keyword from before is really just a shortcut for "when" + "then".
     *  Similar as before Spock knows where the assertion takes place.
     *  The statements in the "then" block will automatically be asserted.
     *
     *  If you try to mess up the order of your blocks then Spock will not allow
     *  your tests to be compiled and run!
     *  Yes you read this correctly, Spock ensures that features follow the BDD convention!
     */
    def 'Should be able to remove from list'() {
        given: 'We have a simple list!'
            var list = [1, 2, 3, 4]

        when: 'We remove the first item...'
            list.remove(0)

        then: '...the following should happen:'
            list == [2, 3, 4]
    }

    /*
        The above example also contains documentation for each block label!
        This feature allows us to document the code.
        One might wonder: Why not just use regular comments?
        Well it's because Spock (alongside a plugins for it) allows us
        to generate test reports! For example html reports.
     */

    /**
     *  One easy win for Spock when compared to JUnit is how it cleanly implements parameterized tests.
     *  This is why in the domain of Data Driven Testing Spock truly shines compared to its competitors.
     *  Now, let's look at an example where we use Spocks infamous Data Tables,
     *  which provides a far more convenient way of performing a parameterized test:
     */
    def 'Numbers to the power of two with a fancy data table!'(
            int a, int b, int c
    ) {
        expect: 'Math works!'
            Math.pow(a, b) == c

        where: 'We use the following data:'
            a | b | c
            1 | 2 | 1
            2 | 2 | 4
            3 | 2 | 9
    }

}

