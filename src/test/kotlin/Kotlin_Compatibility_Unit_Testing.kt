
import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
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
        assert( a.toString().equals( "(3x2):[0.0+0.0i, 0.0+1.0i, 1.0+0.0i, 1.0+1.0i, 2.0+0.0i, 2.0+1.0i]" ) )
        assert( b.toString().equals( "(3x2):[0.0+0.0i, 1.0+0.0i, 0.0+1.0i, 1.0+1.0i, 0.0+2.0i, 1.0+2.0i]" ) )
        assert( !a.isVirtual() )
        assert( !b.isVirtual() )
        assert( (a+b).toString().equals( "(3x2):[0.0+0.0i, 1.0+1.0i, 1.0+1.0i, 2.0+2.0i, 2.0+2.0i, 3.0+3.0i]" ) )
        assert( (a-b).toString().equals( "(3x2):[0.0+0.0i, -1.0+1.0i, 1.0-1.0i, 0.0+0.0i, 2.0-2.0i, 1.0-1.0i]" ) )
        assert( (a*b).toString().equals( "(3x2):[0.0+0.0i, 0.0+1.0i, 0.0+1.0i, 0.0+2.0i, 0.0+4.0i, 0.0+5.0i]" ) )
    }

    @Test
    fun optimization_is_being_called() {

        val t : Tsr<Double> = Tsr.of(2.0).setRqsGradient(true)

        t.set(
            Optimizer.of({ w -> w - 4.0 })
        )

        t.backward()

        assert(t.toString().equals("(1):[2.0]:g:[1.0]"))

        t.applyGradient()

        assert(t.toString().equals("(1):[-1.0]:g:[null]"))

    }

}