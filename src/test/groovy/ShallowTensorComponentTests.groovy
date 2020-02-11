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
                "Axis One" : ["1", "2"],
                "Axis Two" : null,
                "Axis Three" : [1, 2]
        ])
        asString = t.index().toString()

        assert !asString.contains(" a ")
        assert !asString.contains(" b ")
        assert !asString.contains(" c ")
        assert asString.contains("1")

        assert asString.contains("Axis One")
        assert asString.contains("Axis Two")
        assert asString.contains("Axis Three")

        assert asString.contains("|     Axis One     |     Axis Two     |    Axis Three    |")


        //print(t.index().toString())


    }








}
