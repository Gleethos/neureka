package it.framing

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

import java.lang.ref.WeakReference

class Tensor_Framing_Integration_Tests extends Specification
{

    def 'Added labels to tensors are accessible through the "index()" method.'()
    {
        given :
            Tsr t = new Tsr([2, 3, 2], 1..100)
            t.label([
                    ["1", "2"],
                    ["a", "b", "c"],
                    [1, 2]
            ])
            String asString = t.index().toString()

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
            asString = t.index().toString()

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
            asString = t.index().toString()

        then :
            t.index().keysOf("Axis Three", 0).contains("tim")
            t.index().keysOf("Axis Three", 1).contains("tina")
            t.index().keysOf("Axis One").contains("x")
            t.index().keysOf("Axis One").contains("y")
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
            t.index().replace("Axis Two", 1, "Hello")
            asString = t.index().toString()

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

    void 'Tensors can be labeled and their labels can be used to extract slices / subsets of tensors.'(){

        given :
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
        Neureka.instance().settings().view().setIsUsingLegacyView(true)
        Tsr t = new Tsr([3, 4],[
                1, 2, 3, 4,
                9, 8, 6, 5,
                4, 5, 6, 7
        ])
        t.label([
                ["1", "2", "3"],
                ["a", "b", "y", "z"]
        ])

        when : Tsr x = t["2", 1..2]
        then :
            x in t
            t.contains(x)
            x.toString().contains("[1x2]:(8.0, 6.0)")

        when : x = t.getAt("2", new int[]{1,2}) // x.toString(): "(1x2):[8.0, 6.0]"
        then :
            x in t
            t.contains(x)
            x.toString().contains("[1x2]:(8.0, 6.0)")

        when : x = t["2".."3", "b".."y"]
        then :
            x in t
            t.contains(x)
            x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")

        when : x = t.getAt(new String[]{"2","3"}, new String[]{"b","y"}) // x.toString(): "(2x2):[8.0, 6.0, 5.0, 6.0]"
        then :
            x in t
            t.contains(x)
            x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")

        when :
            t = new Tsr([2, 3, 4], 7)
            t.label([
                    ["1", "2"],
                    ["a", "b", "y"],
                    ["tim", "tom", "tina", "tanya"]
            ])
            x = t["2", "b".."y", [["tim","tanya"]:2]]

        then :
            x in t
            t.contains(x)
            x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
            x.isVirtual()
            x.isSlice()
            t.isSliceParent()

        when : x = t["2", [["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            x in t
            t.contains(x)
            x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
            x.isVirtual()
            x.isSlice()
            t.isSliceParent()

            t.sliceCount()==2

        when : x = t[[["2"]:1, ["b".."y"]:1, ["tim","tanya"]:2]]
        then :
            x in t
            t.contains(x)
            x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
            x.isVirtual()
            x.isSlice()
            t.isSliceParent()
            t.sliceCount()==3

        when :
        t.label(
                new String[][]{
                        new String[]{"1", "2"},
                        new String[]{"a", "b", "y"},
                        new String[]{"tim", "tom", "tina", "tanya"}
                }
        )
        x = t[["1","2"], "b".."y", [["tim","tanya"]:2]]

        then :
            x in t
            t.contains(x)
            x.toString().contains("[2x2x2]:(7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0)")
            x.isVirtual()
            x.isSlice()
            t.isSliceParent()
            t.sliceCount()==4

        when : '...we make the GC collect some garbage...'
            WeakReference weak = new WeakReference(x)
            x = null
            System.gc()
            for(int i : 1..100){
                if(weak.get()==null) break
                Thread.sleep(10)
            }

        then : 'The weak reference is null because the tensor had no string reference to it! (No memory leak!)'
            weak.get()!=null

    }




}
