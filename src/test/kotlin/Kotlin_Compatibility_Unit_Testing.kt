
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
        Neureka.get().settings().view().tensors {
            it.isScientific = true
            it.isMultiline = false
            it.hasGradient = true
            it.cellSize = 1
            it.hasValue = true
            it.hasRecursiveGraph = false
            it.hasDerivatives = true
            it.hasShape = true
            it.isCellBound = false
            it.postfix = ""
            it.prefix = ""
            it.hasSlimNumbers = false
        }
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
    fun we_can_use_the_subscription_operator_to_slice_tensors() {

        // Given :
        val t : Tsr<ComplexNumber> = Tsr.of(
                        DataType.of(ComplexNumber::class.java),
                        intArrayOf(2, 3, 4),
                        { _, i -> ComplexNumber( i[0].toDouble(), i[1].toDouble() + i[2].toDouble()/10 ) }
                    )

        // When :
        val e1 = t[1, 2, 0]
        val e2 = t[0, 1, 2]
        val e3 = t[16]

        // Then :
        assert(e1.toString() == "(1x1x1):[1.0+2.0i]")
        assert(e2.toString() == "(1x1x1):[0.0+1.2i]")
        assert(e3.toString() == "(1x1x1):[1.0+1.0i]")

        // When :
        val slice = t[0..1, 2..2, 1..3]

        // Then :
        assert(
            t.toString({ it.isMultiline = true }).trim() == """
            (2x3x4):[
               [
                  [ 0.0+0.0i, 0.0+0.1i, 0.0+0.2i, 0.0+0.3i ],
                  [ 0.0+1.0i, 0.0+1.1i, 0.0+1.2i, 0.0+1.3i ],
                  [ 0.0+2.0i, 0.0+2.1i, 0.0+2.2i, 0.0+2.3i ]
               ],
               [
                  [ 1.0+0.0i, 1.0+0.1i, 1.0+0.2i, 1.0+0.3i ],
                  [ 1.0+1.0i, 1.0+1.1i, 1.0+1.2i, 1.0+1.3i ],
                  [ 1.0+2.0i, 1.0+2.1i, 1.0+2.2i, 1.0+2.3i ]
               ]
            ]
            """
            .trimIndent()
        )
        // And :
        assert(
            slice.toString({ it.isMultiline = true }) ==
            """
                (2x1x3):[
                   [
                      [ 0.0+2.1i, 0.0+2.2i, 0.0+2.3i ]
                   ],
                   [
                      [ 1.0+2.1i, 1.0+2.2i, 1.0+2.3i ]
                   ]
                ]
            """
            .trimIndent()
        )

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
    fun settings_API_for_formatting_tensors_is_convenient_in_kotlin() {

        // Given:
        val viewSettings = Neureka.get().settings().view()
        // And :
        viewSettings.tensors {
                        it.isCellBound = true
                        it.hasSlimNumbers = false
                        it.prefix = "§"
                        it.postfix = "!"
                        it.isMultiline = true
                        it.isScientific = true
                        it.isLegacy = true
                        it.hasGradient = true
                        it.hasShape = false
                        it.cellSize = 8
                        it.rowLimit = 30
                    }
        // When :
        val t = Tsr.of(Float::class.java)
                        .withShape(2, 4, 3)
                        .andWhere { i, index -> index.sum().toFloat()/i  }

        // Then :
        assert(t.toString() == """
                                        §:(
                                           (
                                              (    NaN  ,    1.0  ,    1.0   ),
                                              ( 0.3333..,    0.5  , 0.6000.. ),
                                              ( 0.3333.., 0.4285..,    0.5   ),
                                              ( 0.3333.., 0.4000.., 0.4545.. )
                                           ),
                                           (
                                              ( 0.0833.., 0.1538.., 0.2142.. ),
                                              ( 0.1333..,  0.1875 , 0.2352.. ),
                                              ( 0.1666.., 0.2105..,   0.25   ),
                                              ( 0.1904.., 0.2272.., 0.2608.. )
                                           )
                                        )!
                                """.trimIndent())

        // When :
        viewSettings.tensors {
            it.isCellBound = true
            it.hasSlimNumbers = true
            it.prefix = ".."
            it.postfix = "°°"
            it.isMultiline = false
            it.isScientific = true
            it.isLegacy = false
            it.hasGradient = true
            it.hasShape = true
            it.cellSize = 4
            it.rowLimit = 6
        }

        // Then :
        assert(t.toString() == "..(2x4x3):[ NaN,   1 ,   1 , .3..,  .5 , .6.., ... + 18 more]°°")

        // When :
        Neureka.get().reset()

        // Then :
        assert(t.toString() == """
            (2x4x3):[
               [
                  [   NaN ,   1.0 ,   1.0  ],
                  [ 0.33333E0,   0.5 , 0.60000E0 ],
                  [ 0.33333E0, 0.42857E0,   0.5  ],
                  [ 0.33333E0, 0.40000E0, 0.45454E0 ]
               ],
               [
                  [ 0.08333E0, 0.15384E0, 0.21428E0 ],
                  [ 0.13333E0, 0.1875, 0.23529E0 ],
                  [ 0.16666E0, 0.21052E0,  0.25  ],
                  [ 0.19047E0, 0.22727E0, 0.26086E0 ]
               ]
            ]""".trimIndent())
    }


    @Test
    fun settings_API_for_formatting_tensors_allows_us_to_configure_the_indent() {

        // Given:
        val viewSettings = Neureka.get().settings().view()
        // And :
        viewSettings.tensors {
            it.isCellBound = true
            it.hasSlimNumbers = true
            it.isMultiline = true
            it.prefix = "START\n"
            it.postfix = "\nEND"
            it.isScientific = true
            it.isLegacy = false
            it.hasGradient = true
            it.hasShape = true
            it.cellSize = 6
            it.rowLimit = 5
            it.hasDerivatives = true
            it.hasRecursiveGraph = true
        }
        // When :
        val t = Tsr.of(Double::class.java)
                    .withShape(2, 3, 9)
                    .andWhere { i, index -> index.sum().toDouble()/i  }
                    .setRqsGradient(true)

        // And :
        val o = t * Tsr.of(6.0)

        // And :
        o.backward(2.0)

        // Then :
        assert(t.toString() == """
                                    START
                                    (2x3x9):[
                                       [
                                          [   NaN ,    1  ,    1  , ..4 more..,    1  ,    1   ],
                                          [ .11111,   .2  , .27272, ..4 more..,   .5  , .52941 ],
                                          [ .11111, .15789,   .2  , ..4 more..,   .36 , .38461 ]
                                       ],
                                       [
                                          [ .03703, .07142, .10344, ..4 more.., .23529, .25714 ],
                                          [ .05555, .08108, .10526, ..4 more.., .20930, .22727 ],
                                          [ .06666, .08695, .10638, ..4 more.., .19230, .20754 ]
                                       ]
                                    ]:g:[
                                       [
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ],
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ],
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ]
                                       ],
                                       [
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ],
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ],
                                          [   12  ,   12  ,   12  , ..4 more..,   12  ,   12   ]
                                       ]
                                    ]
                                    END
                                """.trimIndent())
    }

}