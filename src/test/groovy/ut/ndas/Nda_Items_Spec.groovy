package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The Nds Items API")
@Narrative('''

    Nd-arrays are collections of items similar to other
    collection types in Java. 
    One useful way to access the items of an nd-array is
    to use the items API.
    
    Using the `at` methods we can access an `Item` object
    which is a wrapper around the item's value and its
    index in the nd-array.
    
    The `Item` object is a simple data class which
    is very similar to the `Optional` class, meaning
    that it can either be empty or contain a value.

''')
class Nda_Items_Spec extends Specification
{
    def 'We can check if items of a tensor is present or not.'()
    {
        given : 'An nd-array with some data.'
            var nda = Nda.of( "9", "42", null, "3" )
        expect : 'The items are present or not as expected.'
            nda.at(0).exists()
            nda.at(1).exists()
            nda.at(2).doesNotExist()
            nda.at(3).exists()
    }

    def 'We can get the value of an item.'()
    {
        given : 'An nd-array with some data.'
            var nda = Nda.of( "73", null, "42", "3" )
        expect : 'The items can be accessed by their index.'
            nda.at(0).orElseNull() == "73"
            nda.at(1).orElseNull() == null
            nda.at(2).orElseNull() == "42"
            nda.at(3).orElseNull() == "3"
    }

    def 'The "get" method of an Item object will throw an exception if the item is missing.'()
    {
        reportInfo """
            Similar as the `Optional` class, an `Item` object can be empty.
            If we try to get the value of an empty item, an exception will be thrown.
            The reason for this is that we can not be sure that the item is actually
            empty or if it is just not present in the nd-array.
            If you want to get an item's value without throwing an exception 
            (but the risk of getting a null value instead) you can use the `orElseNull` method.
        """
        given : 'An nd-array with some data.'
            var nda = Nda.of( "a", null, "c" )
        expect : 'The non null items can be accessed by their index.'
            nda.at(0).get() == "a"
            nda.at(2).get() == "c"
        when : 'We try to get the value of an empty item.'
            nda.at(1).get()
        then : 'An exception is thrown.'
            thrown( Exception )
    }

    def 'We can use the "orElse(T)" method to avoid null values.'()
    {
        given : 'An nd-array with some data.'
            var nda = Nda.of( "x", null, "z" )
        expect : 'The non null items can be accessed by their index, and the provided value is ignored.'
            nda.at(0).orElse("y") == "x"
            nda.at(2).orElse("y") == "z"
        and : 'When we try to get the value of an empty item we get the provided value instead.'
            nda.at(1).orElse("y") == "y"
    }

    def 'An item can be converted to an Optional object.'()
    {
        given : 'An nd-array with some data.'
            var nda = Nda.of( "a", null, "c" )
        expect : 'The items can be converted to Optional objects.'
            nda.at(0).toOptional().get() == "a"
            nda.at(1).toOptional().orElse(null) == null
            nda.at(2).toOptional().get() == "c"
    }

    def 'Other than the "orElse(T)" method of the Optional class, the same method of an Item will throw an exception if the provided value is null.'()
    {
        reportInfo """
            If you want to get an item's value without throwing an exception 
            (but the risk of getting a null value instead) you can use the `orElseNull` method.
            The `orElse(T)` method of the `Optional` class will not throw an exception
            if the provided value is null. This is not the case for the `orElse(T)` method
            of an `Item` object.
        """
        given : 'An nd-array with some data.'
            var nda = Nda.of( "a", null, "c" )
        expect : 'The items can be converted to Optional objects.'
            nda.at(0).orElse("b") == "a"
            nda.at(1).orElse("b") == "b"
            nda.at(2).orElse("b") == "c"
        when : 'We try to get the value of an empty item.'
            nda.at(1).orElse(null)
        then : 'An exception is thrown.'
            thrown( Exception )
    }
}
