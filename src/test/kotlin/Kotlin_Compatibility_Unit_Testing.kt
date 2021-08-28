
import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.calculus.args.Arg
import neureka.calculus.args.Args
import neureka.dtype.DataType
import neureka.optimization.Optimizer
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test


internal class Kotlin_Compatibility_Unit_Testing {

    @BeforeEach
    fun setupSpec()
    {
        Neureka.get().reset()

        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().setAsString("dgc")
    }

    @Test
    fun operator_overloading_works_for_scalars_in_kotlin() {

        listOf(
            Pair(-4.0, { t1 : Tsr<Double>, t2 : Tsr<Double> -> t1 * t2 }),
            Pair( 3.0, { t1 : Tsr<Double>, t2 : Tsr<Double> -> t1 + t2 }),
            Pair(-4.0, { t1 : Tsr<Double>, t2 : Tsr<Double> -> t1 / t2 }),
            Pair( 5.0, { t1 : Tsr<Double>, t2 : Tsr<Double> -> t1 - t2 })
        )
        .forEach { pair ->

            val expected = pair.first
            val exec     = pair.second

            val t1 : Tsr<Double> = Tsr.of( 4.0  ).setRqsGradient(true)
            val t2 : Tsr<Double> = Tsr.of( -1.0 )

            // When:
            val t3 = exec(t1, t2)

            // Then:
            assert( t3.size() == 1 )

            // And:
            assert( t3.toList().all { it == expected } )

            // When:
            val graphNode = t3[GraphNode::class.java]

            // Then:
            assert( graphNode != null )
            assert( graphNode.usesAD() )

        }
    }

    @Test
    fun tensor_operations_translate_to_custom_ComplexNumber_type_written_in_kotlin()
    {

        // Given :
        val a : Tsr<ComplexNumber> = Tsr.of(
                                        DataType.of(ComplexNumber::class.java),
                                        intArrayOf(3, 2),
                                        { i : Int, indices : IntArray -> ComplexNumber( indices[0].toDouble(), indices[1].toDouble() ) }
                                    )
        val b : Tsr<ComplexNumber> = Tsr.of(
                                        DataType.of(ComplexNumber::class.java),
                                        intArrayOf(3, 2),
                                        { i : Int, indices : IntArray -> ComplexNumber( indices[1].toDouble(), indices[0].toDouble() ) }
                                    )

        // Expect:
        assert(a.toString() == "(3x2):[0.0+0.0i, 0.0+1.0i, 1.0+0.0i, 1.0+1.0i, 2.0+0.0i, 2.0+1.0i]")
        assert(b.toString() == "(3x2):[0.0+0.0i, 1.0+0.0i, 0.0+1.0i, 1.0+1.0i, 0.0+2.0i, 1.0+2.0i]")
        assert( !a.isVirtual() )
        assert( !b.isVirtual() )
        assert((a+b).toString() == "(3x2):[0.0+0.0i, 1.0+1.0i, 1.0+1.0i, 2.0+2.0i, 2.0+2.0i, 3.0+3.0i]")
        assert((a-b).toString() == "(3x2):[0.0+0.0i, -1.0+1.0i, 1.0-1.0i, 0.0+0.0i, 2.0-2.0i, 1.0-1.0i]")
        assert((a*b).toString() == "(3x2):[0.0+0.0i, 0.0+1.0i, 0.0+1.0i, 0.0+2.0i, 0.0+4.0i, 0.0+5.0i]")
    }

    @Test
    fun optimization_is_being_called() {

        listOf(
            Pair( -3.0, { g : Tsr<Double> -> g - 4.0 } ), // 'g' will always be 1
            Pair(  5.0, { g : Tsr<Double> -> g + 4.0 } ),
            Pair( 0.25, { g : Tsr<Double> -> g / 4.0 } ),
            Pair(  4.0, { g : Tsr<Double> -> g * 4.0 } ),
            Pair(  0.0, { g : Tsr<Double> -> g % 1   } )
        )
        .forEach { pair ->

            // Given :
            val expected = pair.first
            val exec = pair.second
            val weightVal = 2.0
            val w: Tsr<Double> = Tsr.of(weightVal).setRqsGradient(true)

            // When :
            w.set( Optimizer.ofGradient( { g -> exec(g) } ) ).backward()

            // Then :
            assert(w.toString() == "(1):["+weightVal+"]:g:[1.0]")

            // When :
            w.applyGradient()

            // Then :
            assert(w.toString() == "(1):["+(expected+weightVal)+"]:g:[null]")
        }
    }

    @Test
    fun convenience_methods_in_function_API_are_consistent() {

        listOf(
            Pair( "(1):[4.0]", { Function.of("i0 * 4 - 3").with(Arg.DerivIdx.of(0))(Tsr.of(5.0)) } ),
            Pair( "(1):[6.0]", { Function.of("i0 * i0").execute(Args.of(Arg.DerivIdx.of(0)), Tsr.of(3.0)) } )
        )
        .forEach { pair ->
            // Given :
            val expected = pair.first
            val exec = pair.second

            // When :
            val t = exec()

            // Then :
            assert(t.toString() == expected)
        }
    }

}