import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.calculus.Function
import org.junit.Test

class ADMemTests {

    @Test
    void test_reverse_reshape() {

        Neureka.instance().settings().reset()
        Neureka.instance().settings().view().setLegacy(true)

        Tsr a = new Tsr([2, 3], [
                1, 2, 3,
                4, 5, 6
        ]).setRqsGradient(true)

        Function rs = Function.create("[1, 0]:(I[0])")

        Tsr b = rs.activate(a)
        GraphNode n = a.find(GraphNode.class)
        assert n.getChildren().size()==1

        assert  b.toString().contains("")

        b.backward(new Tsr([3, 2],[
                -1, 2,
                4, 7,
                -9, 8
        ]))
        assert a.toString().contains("[2x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0):g:(-1.0, 4.0, -9.0, 2.0, 7.0, 8.0)")
        assert b.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")

        GraphNode node = a.find(GraphNode.class)
        assert node.isLeave()
        assert node.function()==null
        assert node.mode() == 1
        assert node.size()==0
        assert node.nid()==1
        assert node.lock().isLocked()==false

        node = b.find(GraphNode.class)
        assert !node.isLeave()
        assert node.function()!=null
        assert node.mode() == -1
        assert node.size()==0
        assert node.nid()!=1
        assert !node.lock().isLocked()

    }

    @Test
    void test_payloads_and_derivatives_are_null() {

        Neureka.instance().settings().reset()

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = a * 3 / 5
        Tsr c = b ^ new Tsr(3)
        Tsr d = c / 100
        GraphNode n = d.find(GraphNode.class)

        assert n.parents[0].isCachable()
        assert !n.parents[0].isLeave()
        assert n.parents[0].isGraphLeave()
        assert n.parents[1].isLeave()
        assert n.parents[1].isGraphLeave()

        for (int i = 0; i < n.parents.length; i++) {
            assert n.parents[i].payload != null
            boolean[] exists = {false}
            n.parents[i].forEachTarget({ t -> exists[0] = true })
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

        for (int i = 0; i < n.parents.length; i++) {
            assert n.parents[i].payload == null
            assert !n.parents[i].hasDerivatives()
        }

    }


}
