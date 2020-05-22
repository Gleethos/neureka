import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import org.junit.Test

class ShallowTensorComponentTests {

    @Test
    void test_all_operations(){

        Neureka.instance().reset()
        Tsr t = new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])

        Function f = Function.create("relu(I[0])")
        Tsr o = f(t)
        assert o.toString().contains("(2x3):[-0.04, 7.0, -0.01, 2.0, 3.0, 8.0]")
        o = f.derive(new Tsr[]{t}, 0)
        assert o.toString().contains("(2x3):[0.01, 1.0, 0.01, 1.0, 1.0, 1.0]")

        f = Function.create("quad(I[0])")
        o = f(t)
        assert o.toString().contains("(2x3):[16.0, 49.0, 1.0, 4.0, 9.0, 64.0]")
        o = f.derive(new Tsr[]{t}, 0)
        assert o.toString().contains("(2x3):[-8.0, 14.0, -2.0, 4.0, 6.0, 16.0]")

        f = Function.create("abs(I[0])")
        o = f(t)
        assert o.toString().contains("(2x3):[4.0, 7.0, 1.0, 2.0, 3.0, 8.0]")
        o = f.derive(new Tsr[]{t}, 0)
        assert o.toString().contains("(2x3):[-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]")
    }

    @Test
    void test_custom_index()
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
