
import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.backend.api.Call
import neureka.calculus.Function
import neureka.calculus.args.Arg
import neureka.calculus.args.Args
import neureka.devices.host.CPU
import neureka.dtype.DataType
import neureka.optimization.Optimizer
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test


internal class
Kotlin_Compatibility_Unit_Testing {

    @BeforeEach
    fun setupSpec()
    {
        Neureka.get().reset()

        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors("dgc")
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
                                        { _ : Int, indices : IntArray -> ComplexNumber( indices[0].toDouble(), indices[1].toDouble() ) }
                                    )
        val b : Tsr<ComplexNumber> = Tsr.of(
                                        DataType.of(ComplexNumber::class.java),
                                        intArrayOf(3, 2),
                                        { _ : Int, indices : IntArray -> ComplexNumber( indices[1].toDouble(), indices[0].toDouble() ) }
                                    )

        // Expect:
        assert(a.toString() == "(3x2):[0.0+0.0i, 0.0+1.0i, 1.0+0.0i, 1.0+1.0i, 2.0+0.0i, 2.0+1.0i]")
        assert(b.toString() == "(3x2):[0.0+0.0i, 1.0+0.0i, 0.0+1.0i, 1.0+1.0i, 0.0+2.0i, 1.0+2.0i]")
        assert( !a.isVirtual )
        assert( !b.isVirtual )
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
            assert(w.toString() == "(1):[$weightVal]:g:[1.0]")

            // When :
            w.applyGradient()

            // Then :
            assert(w.toString() == "(1):["+(expected+weightVal)+"]:g:[null]")
        }
    }

    @Test
    fun convenience_methods_in_function_API_are_consistent() {

        listOf(
            Pair( "(1):[4.0]", { Function.of("i0 * 4 - 3").callWith(Arg.DerivIdx.of(0))(Tsr.of(5.0)) } ),
            Pair( "(1):[4.0]", { Function.of("i0 * 4 - 3").invoke(Call.to(CPU.get()).with(Tsr.of(5.0)).andArgs(Arg.DerivIdx.of(0))) } ),
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

    @Test
    fun settings_API_for_formatting_tensors_is_convinient_in_kotlin() {

        // Given:
        val viewSettings = Neureka.get().settings().view()
        // And :
        viewSettings.tensors {
                        it.cellBound(true)
                            .withSlimNumbers(false)
                            .withPrefix("§")
                            .withPostfix("!")
                            .multiline(true)
                            .scientific(true)
                            .legacy(true)
                            .withGradient(true)
                            .withShape(false)
                            .withPadding(8)
                            .withRowLimit(30)
                }
        // When :
        val t = Tsr.of(Float::class.java)
                        .withShape(2, 4, 3)
                        .andWhere { i, index -> index.sum().toFloat()/i  }

        // Then :
        assert(t.toString() == """
                    §:[
                       [
                          [    NaN  ,    1.0  ,    1.0   ],
                          [ 0.3333..,    0.5  , 0.6000.. ],
                          [ 0.3333.., 0.4285..,    0.5   ],
                          [ 0.3333.., 0.4000.., 0.4545.. ]
                       ],
                       [
                          [ 0.0833.., 0.1538.., 0.2142.. ],
                          [ 0.1333..,  0.1875 , 0.2352.. ],
                          [ 0.1666.., 0.2105..,   0.25   ],
                          [ 0.1904.., 0.2272.., 0.2608.. ]
                       ]
                    ]!
        """.trimIndent())

        // When :
        viewSettings.tensors {
            it.cellBound(true)
                .withSlimNumbers(true)
                .withPrefix("..")
                .withPostfix("°°")
                .multiline(false)
                .scientific(true)
                .legacy(false)
                .withGradient(true)
                .withShape(true)
                .withPadding(4)
                .withRowLimit(6)
        }

        // Then :
        assert(t.toString() == """
                    ..(2x4x3):[ NaN,   1 ,   1 , .3..,  .5 , .6.., ... + 18 more]°°
                """.trimIndent())

        Neureka.get().reset()



    }

}