import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import org.junit.Test

class ADMemTests {


    @Test
    void testPayloadsAndDerivativesAreNull(){

        Neureka.Settings.reset()

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = a*3/5
        Tsr c = b ^ new Tsr(3)
        Tsr d = c / 100
        GraphNode n = d.find(GraphNode.class)

        for(int i=0; i<n.parents.length; i++){
            assert n.parents[i].payload != null
            boolean[] exists = {false}
            n.parents[i].forEach({t, g -> exists[0] = true })
            assert exists[0]
        }
        a = null
        b = null
        c = null
        System.gc()
        Thread.sleep(200)
        System.gc()
        Thread.sleep(200)
        System.gc()

        for(int i=0; i<n.parents.length; i++){
            assert n.parents[i].payload == null
            assert !n.parents[i].hasDerivatives()
        }

    }




}
