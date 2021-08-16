
import neureka.Tsr
import neureka.autograd.GraphNode
import org.junit.jupiter.api.Test


internal class Kotlin_Compatibility_Unit_Testing {

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


}