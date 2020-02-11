import neureka.Tsr
import org.junit.Test

class ShallowTensorComponentTests {

    @Test
    void testIndex()
    {
        Tsr t = new Tsr([2, 3, 2], 1..100)

        t.label([
            ["1", "2"],
            ["a", "b", "c"],
            [1, 2]
        ])

        String asString = t.index().toString()
        assert asString.contains("a")
        assert asString.contains("b")
        assert asString.contains("c")
        assert asString.contains("1")

        assert !asString.contains("Axis One")
        assert !asString.contains("Axis Two")
        assert !asString.contains("Axis Three")

        t.label([
            "Axis One" : ["1", "2"],
            "Axis Two" : ["a", "b", "c"],
            "Axis Three" : [1, 2]
        ])

        asString = t.index().toString()
        assert asString.contains("a")
        assert asString.contains("b")
        assert asString.contains("c")
        assert asString.contains("1")

        assert asString.contains("Axis One")
        assert asString.contains("Axis Two")
        assert asString.contains("Axis Three")

        assert asString.contains("|     Axis One     |     Axis Two     |    Axis Three    |")

        t.label([
                "Axis One" : ["x", "y"],
                "Axis Two" : null,
                "Axis Three" : ["tim", "tina"]
        ])
        asString = t.index().toString()

        assert t.index().keysOf("Axis Three", 0).contains("tim")
        assert t.index().keysOf("Axis Three", 1).contains("tina")
        assert t.index().keysOf("Axis One").contains("x")
        assert t.index().keysOf("Axis One").contains("y")
        assert !asString.contains(" a ")
        assert !asString.contains(" b ")
        assert !asString.contains(" c ")
        assert asString.contains("x")
        assert asString.contains("tim")
        assert asString.contains("tina")
        assert asString.contains("0")
        assert asString.contains("1")
        assert asString.contains("2")

        assert asString.contains("Axis One")
        assert asString.contains("Axis Two")
        assert asString.contains("Axis Three")

        assert asString.contains("|     Axis One     |     Axis Two     |    Axis Three    |")

        t.index().replace("Axis Two", 1, "Hello")

        asString = t.index().toString()

        assert !asString.contains(" a ")
        assert !asString.contains(" b ")
        assert !asString.contains(" c ")
        assert asString.contains("x")
        assert asString.contains("tim")
        assert asString.contains("tina")
        assert asString.contains("0")
        assert !asString.contains("1")
        assert asString.contains("Hello")
        assert asString.contains("2")

        print(t.index().toString())


    }








}
