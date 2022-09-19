package ut.framing


import neureka.Neureka
import neureka.Tsr
import neureka.framing.NDFrame
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title
import testutility.Sleep

import java.lang.ref.WeakReference

@Title("Naming Tensors and their Dimensions.")
@Narrative('''

    A powerful concept in the data science as well as machine learning
    world is something usually referred to as "Data Frames".
    These are highly flexible 2D data structures
    used to load and store CSV, CRV, etc... files for 
    data exploration and further processing.
    Data frames are so powerful because
    their indices are labeled and therefore human readable.
    Neureka's tensors are general purpose data containers
    which may also stored data in 2 dimensions whose
    indices may also be something other than integers.

''')
@Subject([Tsr, NDFrame])
class Tensor_Framing_Spec extends Specification
{
    def setupSpec() {
        reportHeader """
                    This specification covers the behavior
                    of tensors with respect to specifying aliases for
                    indices and then using them for slicing.     
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

    def 'We can add labels to tensors through lists or maps passed to the "label()" method.'()
    {
        given : 'We create a 3D tensor and label its indices.'
            Tsr t = Tsr.of([2, 3, 2], 1..100)
            t.label([
                    ["1", "2"],
                    ["a", "b", "c"],
                    [1, 2]
            ])
            String asString = t.frame().toString()

        expect : 'The string representation of the tensor should include these labels.'
            asString.contains("a")
            asString.contains("b")
            asString.contains("c")
            asString.contains("1")

            !asString.contains("Axis One")
            !asString.contains("Axis Two")
            !asString.contains("Axis Three")

        when : 'We provide a map, where the keys are axis labels...'
            t.label([
                    "Axis One" : ["1", "2"],
                    "Axis Two" : ["a", "b", "c"],
                    "Axis Three" : [1, 2]
            ])
            asString = t.frame().toString()

        then : 'Once again, the string will mention all labels'
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
            t.frame().atAxis("Axis Three").getAllAliasesForIndex(0) == ["tim"]
            t.frame().atAxis("Axis Three").getAllAliasesForIndex(1) == ["tina"]
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

    void 'A matrix (rank 2 tensor) can be labeled and their labels can be used to extract slices / subsets.'()
    {
        given: 'Tensor printing is set to "legacy" for this test.'
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
        and: 'A labeled tensor of rank 2 is being created.'
            Tsr t = Tsr.of([3, 4], [
                    1d, 2d, 3d, 4d,
                    9d, 8d, 6d, 5d,
                    4d, 5d, 6d, 7d
            ])
            t.label([
                    ["1", "2", "3"],
                    ["a", "b", "y", "z"]
            ])

        expect:
            t.toString({
                        it.rowLimit = 15
                        it.isScientific = false
                        it.isMultiline = true
                        it.hasGradient = false
                        it.cellSize = 6
                        it.hasValue = true
                        it.hasRecursiveGraph = false
                        it.hasDerivatives = false
                        it.hasShape =  true
                        it.isCellBound = false
                        it.postfix = ""
                        it.prefix = ""
                        it.hasSlimNumbers = false
            }) == "[3x4]:(\n" +
                  "   [    a  ][   b  ][   y  ][   z   ]\n" +
                  "   (   1.0 ,   2.0 ,   3.0 ,   4.0  ):[ 1 ],\n" +
                  "   (   9.0 ,   8.0 ,   6.0 ,   5.0  ):[ 2 ],\n" +
                  "   (   4.0 ,   5.0 ,   6.0 ,   7.0  ):[ 3 ]\n" +
                  ")"

        when: 'We use a label for slicing a row from the tensor (which is also a matrix in this case).'
            Tsr x = t["2", 1..2]
        then: 'This new slice "x" then will yield true when using the "contains" operator on t.'
            x in t
        and: 'Calling the "contains" method will also return true.'
            t.contains(x)
        and: 'The String representation is as expected.'
            x.toString().contains("[1x2]:(8.0, 6.0)")
        and: 'The tensor "x" is of course a (partial) slice:'
            x.isSlice()
            x.isPartialSlice()
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
            x.isPartialSlice()
            t.isSliceParent()

        when: 'Calling the "getAt" method manually with "ranges" as String arrays...'
            x = t.getAt(new String[]{"2", "3"}, new String[]{"b", "y"}) // x.toString(): "(2x2):[8.0, 6.0, 5.0, 6.0]"
        then: '...this will produce the same result as previously:'
            x in t
            t.contains(x)
            x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")
            x.isSlice()
            x.isPartialSlice()
            t.isSliceParent()
    }

    def 'Rank 3 tensors can be labeled and their labels can be used to extract slices / subsets of tensors.' ()
    {
        given: 'Tensor printing is set to "legacy" for this test.'
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
        and: 'A labeled tensor of rank 3 is being created.'
            var t = Tsr.of([2, 3, 4], -7d..7d)
            t.label( 'My Tensor', [
                ["1", "2"],
                ["a", "b", "y"],
                ["tim", "tom", "tina", "tanya"]
            ])

        expect: 'When the tensor is converted to a String then the labels will be included:'
            t.toString({
                it.rowLimit = 15
                it.isScientific = false
                it.isMultiline = true
                it.hasGradient = false
                it.cellSize = 6
                it.hasValue = true
                it.hasRecursiveGraph = false
                it.hasDerivatives = false
                it.hasShape =  true
                it.isCellBound = false
                it.postfix = ""
                it.prefix = ""
                it.hasSlimNumbers = false
            }) == "[2x3x4]:(\n" +
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
                  ")"

        when : 'Creating a slice by passing a single label, a range of labels and a range with stride...'
            var s = t["2", "b".."y", [["tim","tanya"]:2]]

        then : 'This new slice "x" then will yield true when using the "contains" operator on t.'
            s in t
        and: 'Calling the "contains" method will also return true.'
            t.contains(s)
        and: 'The String representation is as expected.'
            s.toString({
                it.rowLimit = 15
                it.isScientific = false
                it.isMultiline = true
                it.hasGradient = false
                it.cellSize = 6
                it.hasValue = true
                it.hasRecursiveGraph = false
                it.hasDerivatives = false
                it.hasShape =  true
                it.isCellBound = false
                it.postfix = ""
                it.prefix = ""
                it.hasSlimNumbers = false
            }) == "[1x2x2]:(\n" +
                  "   (\n" +
                  "      (  -6.0 ,  -4.0  ),\n" +
                  "      (  -2.0 ,   0.0  )\n" +
                  "   )\n" +
                  ")"
        and: 'The tensor "x" is of course a slice:'
            s.isSlice()
        and: 'The original tensor "t" is a "parent":'
            t.isSliceParent()
        and : 'The slice is not virtual.'
            !s.isVirtual() // This might change if possible (technically difficult)


        when : 'We slice the tensor t by passing a map of start and end labels as keys and strides as values.'
            s = t["2", [["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            s in t
            t.contains(s)
            s.toString({
                it.rowLimit = 15
                it.isScientific = false
                it.isMultiline = true
                it.hasGradient = false
                it.cellSize = 6
                it.hasValue = true
                it.hasRecursiveGraph = false
                it.hasDerivatives = false
                it.hasShape =  true
                it.isCellBound = false
                it.postfix = ""
                it.prefix = ""
                it.hasSlimNumbers = false
            }) == "[1x2x2]:(\n" +
                  "   (\n" +
                  "      (  -6.0 ,  -4.0  ),\n" +
                  "      (  -2.0 ,   0.0  )\n" +
                  "   )\n" +
                  ")"
            !s.isVirtual() // This might change if possible (technically difficult)
            s.isSlice()
            t.isSliceParent()
            t.sliceCount()==2

        when : 'We slice the tensor t by passing a map of start and end labels as keys and strides as values.'
            s = t[[["2"]:1, ["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            s in t
            t.contains(s)
            s.toString({
                it.rowLimit = 15
                it.isScientific = false
                it.isMultiline = true
                it.hasGradient = false
                it.cellSize = 6
                it.hasValue = true
                it.hasRecursiveGraph = false
                it.hasDerivatives = false
                it.hasShape =  true
                it.isCellBound = false
                it.postfix = ""
                it.prefix = ""
                it.hasSlimNumbers = false
            }) == "[1x2x2]:(\n" +
                  "   (\n" +
                  "      (  -6.0 ,  -4.0  ),\n" +
                  "      (  -2.0 ,   0.0  )\n" +
                  "   )\n" +
                  ")"
            !s.isVirtual() // This might change if possible (technically difficult)
            s.isSlice()
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
            s = t[ ["1","2"], "b".."y", [["tim","tanya"]:2] ]

        then :
            s in t
            t.contains(s)
            s.toString({
                        it.rowLimit = 15
                        it.isScientific = false
                        it.isMultiline = true
                        it.hasGradient = false
                        it.cellSize = 6
                        it.hasValue = true
                        it.hasRecursiveGraph = false
                        it.hasDerivatives = false
                        it.hasShape =  true
                        it.isCellBound = false
                        it.postfix = ""
                        it.prefix = ""
                        it.hasSlimNumbers = false
                    }) == "[2x2x2]:(\n" +
                          "   (\n" +
                          "      (  -3.0 ,  -1.0  ),\n" +
                          "      (   1.0 ,   3.0  )\n" +
                          "   ),\n" +
                          "   (\n" +
                          "      (  -6.0 ,  -4.0  ),\n" +
                          "      (  -2.0 ,   0.0  )\n" +
                          "   )\n" +
                          ")"
            !s.isVirtual() // This might change if possible (technically difficult)
            s.isSlice()
            t.isSliceParent()
            t.sliceCount() == 4
    /*
        when : '...we make the GC collect some garbage...'
            var weak = new WeakReference(s)
            s = null
            System.gc() // This is not guaranteed to work, but it's the best we can do.
            System.runFinalization() // Idk, maybe this helps.
            Runtime.getRuntime().gc() // Some more attempts to trigger the garbage collection.
            System.runFinalization()

        then : 'The weak reference returns null instead of the slice because the parent has only weak references to it!'
            s == null
            t != null
            Sleep.until(750, { weak.get() == null })
            t.sliceCount() == 0

     */
    }

    def 'A tensor can be labeled partially.'()
    {
        given: 'A labeled tensor of rank 3 is being created.'
            Tsr t = Tsr.of([2, 3, 4], -7d..7d)
            t.label( 'My Tensor', [
                ["1", "2"],
                null, // We don't want to label the rows
                ["tim", "tom"] // We only label 2 of 4
            ])

        expect: 'When the tensor is converted to a String then the specified labels will be included:'
            t.toString({
                it.isMultiline=true; it.isCellBound=true; it.cellSize=7;it.isLegacy=false
            }) == "(2x3x4):[\n" +
                    "   ( 1 ):[\n" +
                    "      (   tim  )(  tom  )(       )(        ):( My Tensor )\n" +
                    "      [   -7.0 ,   -6.0 ,   -5.0 ,   -4.0  ],\n" +
                    "      [   -3.0 ,   -2.0 ,   -1.0 ,   0.0   ],\n" +
                    "      [   1.0  ,   2.0  ,   3.0  ,   4.0   ]\n" +
                    "   ],\n" +
                    "   ( 2 ):[\n" +
                    "      (   tim  )(  tom  )(       )(        ):( My Tensor )\n" +
                    "      [   5.0  ,   6.0  ,   7.0  ,   -7.0  ],\n" +
                    "      [   -6.0 ,   -5.0 ,   -4.0 ,   -3.0  ],\n" +
                    "      [   -2.0 ,   -1.0 ,   0.0  ,   1.0   ]\n" +
                    "   ]\n" +
                    "]"
    }


}
