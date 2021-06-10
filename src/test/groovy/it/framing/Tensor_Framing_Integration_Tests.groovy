package it.framing

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

import java.lang.ref.WeakReference

class Tensor_Framing_Integration_Tests extends Specification
{
    def setupSpec() {
        reportHeader """
                <h2> Framing Behavior </h2>
                <br> 
                <p>
                    This specification covers the behavior
                    of the classes contained in the "framing" package, which 
                    contains logic in order to provide the possibility to alias
                    tensor indices.          
                </p>
            """
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Added labels to tensors are accessible through the "index()" method.'()
    {
        given :
            Tsr t = new Tsr([2, 3, 2], 1..100)
            t.label([
                    ["1", "2"],
                    ["a", "b", "c"],
                    [1, 2]
            ])
            String asString = t.frame().toString()

        expect :
            asString.contains("a")
            asString.contains("b")
            asString.contains("c")
            asString.contains("1")

            !asString.contains("Axis One")
            !asString.contains("Axis Two")
            !asString.contains("Axis Three")

        when :
            t.label([
                    "Axis One" : ["1", "2"],
                    "Axis Two" : ["a", "b", "c"],
                    "Axis Three" : [1, 2]
            ])
            asString = t.frame().toString()

        then :
            asString.contains("a")
            asString.contains("b")
            asString.contains("c")
            asString.contains("1")

            asString.contains("Axis One")
            asString.contains("Axis Two")
            asString.contains("Axis Three")

            asString.contains("|     Axis One     |     Axis Two     |    Axis Three    |")

        when :
            t.label([
                    "Axis One" : ["x", "y"],
                    "Axis Two" : null,
                    "Axis Three" : ["tim", "tina"]
            ])
            asString = t.frame().toString()

        then :
            t.frame().atAxis("Axis Three").getAllAliasesForIndex(0)
            t.frame().atAxis("Axis Three").getAllAliasesForIndex(1)
            t.frame().atAxis("Axis One").getAllAliases().contains("x")
            t.frame().atAxis("Axis One").getAllAliases().contains("y")
            !asString.contains(" a ")
            !asString.contains(" b ")
            !asString.contains(" c ")
            asString.contains("x")
            asString.contains("tim")
            asString.contains("tina")
            asString.contains("0")
            asString.contains("1")
            asString.contains("2")

            asString.contains("Axis One")
            asString.contains("Axis Two")
            asString.contains("Axis Three")

            asString.contains("|     Axis One     |     Axis Two     |    Axis Three    |")

        when :
            //t.index().replace("Axis Two", 1, "Hello")
            t.frame().atAxis("Axis Two").replace(1).with("Hello")
            asString = t.frame().toString()

        then :
            !asString.contains(" a ")
            !asString.contains(" b ")
            !asString.contains(" c ")
            asString.contains("x")
            asString.contains("tim")
            asString.contains("tina")
            asString.contains("0")
            !asString.contains("1")
            asString.contains("Hello")
            asString.contains("2")
    }

    void 'Rank 2 tensors can be labeled and their labels can be used to extract slices / subsets of tensors.'()
    {
        given: 'Tensor printing is set to "legacy" for this test.'
            Neureka.get().settings().view().setIsUsingLegacyView(true)
        and: 'And a labeled tensor of rank 2 is being created.'
            Tsr t = new Tsr([3, 4], [
                    1, 2, 3, 4,
                    9, 8, 6, 5,
                    4, 5, 6, 7
            ])
            t.label([
                    ["1", "2", "3"],
                    ["a", "b", "y", "z"]
            ])

        expect:
            t.toString('fp') == "[3x4]:(\n" +
                                      "   [    a  ][   b  ][   y  ][   z   ]\n" +
                                      "   (   1.0 ,   2.0 ,   3.0 ,   4.0  ):[ 1 ],\n" +
                                      "   (   9.0 ,   8.0 ,   6.0 ,   5.0  ):[ 2 ],\n" +
                                      "   (   4.0 ,   5.0 ,   6.0 ,   7.0  ):[ 3 ]\n" +
                                      ")\n"

        when: 'We use a label for slicing a row from the tensor (which is also a matrix in this case).'
            Tsr x = t["2", 1..2]
        then: 'This new slice "x" then will yield true when using the "contains" operator on t.'
            x in t
        and: 'Calling the "contains" method will also return true.'
            t.contains(x)
        and: 'The String representation is as expected.'
            x.toString().contains("[1x2]:(8.0, 6.0)")
        and: 'The tensor "x" is of course a slice:'
            x.isSlice()
        and: 'The original tensor "t" is a "parent":'
            t.isSliceParent()

        when: 'We now call the "getAt" method manually with the same arguments...'
            x = t.getAt("2", new int[]{ 1, 2 }) // x.toString(): "(1x2):[8.0, 6.0]"
        then: 'This will produce the same result "x" with the same properties...'
            x in t
            t.contains(x)
            x.toString().contains("[1x2]:(8.0, 6.0)")
            x.isSlice()
            t.isSliceParent()

        when: 'Supplying String ranges (whose entries are also labels) for slicing...'
            x = t["2".."3", "b".."y"]
        then: 'This slice will be as expected!'
            x in t
            t.contains(x)
            x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")
            x.isSlice()
            t.isSliceParent()

        when: 'Calling the "getAt" method manually with "ranges" as String arrays...'
            x = t.getAt(new String[]{"2", "3"}, new String[]{"b", "y"}) // x.toString(): "(2x2):[8.0, 6.0, 5.0, 6.0]"
        then: '...this will produce the same result as previously:'
            x in t
            t.contains(x)
            x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")
            x.isSlice()
            t.isSliceParent()
    }

    def 'Rank 3 tensors can be labeled and their labels can be used to extract slices / subsets of tensors.' ()
    {
        given: 'Tensor printing is set to "legacy" for this test.'
            Neureka.get().settings().view().setIsUsingLegacyView(true)
        and: 'And a labeled tensor of rank 3 is being created.'
            Tsr t = new Tsr([2, 3, 4], -7..7)
            t.label( 'My Tensor', [
                ["1", "2"],
                ["a", "b", "y"],
                ["tim", "tom", "tina", "tanya"]
            ])

        expect: 'When the tensor is converted to a String then the labels will be included:'
            t.toString('fp') == "[2x3x4]:(\n" +
                                      "   [ 1 ]:(\n" +
                                      "      [   tim ][  tom ][ tina ][ tanya ]:[ My Tensor ]\n" +
                                      "      (  -7.0 ,  -6.0 ,  -5.0 ,  -4.0  ):[ a ],\n" +
                                      "      (  -3.0 ,  -2.0 ,  -1.0 ,   0.0  ):[ b ],\n" +
                                      "      (   1.0 ,   2.0 ,   3.0 ,   4.0  ):[ y ]\n" +
                                      "   ),\n" +
                                      "   [ 2 ]:(\n" +
                                      "      [   tim ][  tom ][ tina ][ tanya ]:[ My Tensor ]\n" +
                                      "      (   5.0 ,   6.0 ,   7.0 ,  -7.0  ):[ a ],\n" +
                                      "      (  -6.0 ,  -5.0 ,  -4.0 ,  -3.0  ):[ b ],\n" +
                                      "      (  -2.0 ,  -1.0 ,   0.0 ,   1.0  ):[ y ]\n" +
                                      "   )\n" +
                                      ")\n"

        when : 'Creating a slice by passing a single label, a range of labels and a range with stride...'
            Tsr x = t["2", "b".."y", [["tim","tanya"]:2]]

        then : 'This new slice "x" then will yield true when using the "contains" operator on t.'
            x in t
        and: 'Calling the "contains" method will also return true.'
            t.contains(x)
        and: 'The String representation is as expected.'
            x.toString('fp') == "[1x2x2]:(\n" +
                                      "   (\n" +
                                      "      (  -6.0 ,  -4.0  ),\n" +
                                      "      (  -2.0 ,   0.0  )\n" +
                                      "   )\n" +
                                      ")\n"
        and: 'The tensor "x" is of course a slice:'
            x.isSlice()
        and: 'The original tensor "t" is a "parent":'
            t.isSliceParent()
        and : 'The slice is not virtual.'
            !x.isVirtual() // This might change if possible (technically difficult)


        when :
            x = t["2", [["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            x in t
            t.contains(x)
            x.toString('fp') == "[1x2x2]:(\n" +
                                      "   (\n" +
                                      "      (  -6.0 ,  -4.0  ),\n" +
                                      "      (  -2.0 ,   0.0  )\n" +
                                      "   )\n" +
                                      ")\n"
            !x.isVirtual() // This might change if possible (technically difficult)
            x.isSlice()
            t.isSliceParent()
            t.sliceCount()==2

        when : x = t[[["2"]:1, ["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            x in t
            t.contains(x)
            x.toString('fp') == "[1x2x2]:(\n" +
                                      "   (\n" +
                                      "      (  -6.0 ,  -4.0  ),\n" +
                                      "      (  -2.0 ,   0.0  )\n" +
                                      "   )\n" +
                                      ")\n"
            !x.isVirtual() // This might change if possible (technically difficult)
            x.isSlice()
            t.isSliceParent()
            t.sliceCount()==3

        when :
            t.label(
                new String[][]{
                    new String[]{ "1", "2" },
                    new String[]{ "a", "b", "y" },
                    new String[]{ "tim", "tom", "tina", "tanya" }
                }
            )
            x = t[ ["1","2"], "b".."y", [["tim","tanya"]:2] ]

        then :
            x in t
            t.contains(x)
            x.toString('fp') == "[2x2x2]:(\n" +
                                      "   (\n" +
                                      "      (  -3.0 ,  -1.0  ),\n" +
                                      "      (   1.0 ,   3.0  )\n" +
                                      "   ),\n" +
                                      "   (\n" +
                                      "      (  -6.0 ,  -4.0  ),\n" +
                                      "      (  -2.0 ,   0.0  )\n" +
                                      "   )\n" +
                                      ")\n"
            !x.isVirtual() // This might change if possible (technically difficult)
            x.isSlice()
            t.isSliceParent()
            t.sliceCount() == 4

        when : '...we make the GC collect some garbage...'
            WeakReference weak = new WeakReference(x)
            x = null
            System.gc()
            for ( int i : 1..100 ) {
                if( weak.get() == null ) break
                Thread.sleep(10)
            }

        then : 'The weak reference is null because the tensor had no string reference to it! (No memory leak!)'
            weak.get() != null

    }




}
