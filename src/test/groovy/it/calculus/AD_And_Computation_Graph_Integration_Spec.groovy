package it.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.view.TsrStringSettings
import spock.lang.Specification


class AD_And_Computation_Graph_Integration_Spec extends Specification{

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def "Reshaping produces expected computation graph and also works with reverse mode AD."(){

        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr<Double> a = Tsr.of([2, 3], [
                                1d, 2d, 3d,
                                4d, 5d, 6d
                            ]).setRqsGradient(true)
        and :
            Function rs = Function.of("[1, 0]:(I[0])")

        when :
            Tsr b = rs(a)
            GraphNode na = a.get( GraphNode.class )
            GraphNode nb = b.get( GraphNode.class )

        then :
            assert na.getChildren().size()==1
            assert b.toString().contains("")//Todo!
            b.backward(Tsr.of([3, 2],[
                                -1d, 2d,
                                 4d, 7d,
                                -9d, 8d
                        ]))
            assert a.toString().contains("[2x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0):g:(-1.0, 4.0, -9.0, 2.0, 7.0, 8.0)")
            assert b.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")
            assert na.isLeave()
            assert na.function==null
            assert na.getMode() == 1
            assert na.size()==0
            assert na.getNodeID()==1
            assert !na.getLock().isLocked()

            assert !nb.isLeave()
            assert nb.function != null
            assert nb.getMode() == -1
            assert nb.size()==0
            assert nb.getNodeID()!=1
            assert !nb.getLock().isLocked()
    }

    def "Payloads and derivatives are null after garbage collection."()
    {
        given :
            Tsr<Double> a = Tsr.of(2d).setRqsGradient(true)
            Tsr<Double> b = a * 3 / 5
            Tsr<Double> c = Tsr.of(3d)
            Tsr<Double> d = b ^ c
            Tsr<Double> e = d * c
            GraphNode n = e.get( GraphNode.class )

        when : System.gc()
        then :
            assert n.parents[0].isCacheable()
            assert !n.parents[0].isLeave()
            assert n.parents[0].isGraphLeave()
            assert n.parents[1].isLeave()
            assert n.parents[1].isGraphLeave()

        and :
            for ( int i = 0; i < n.parents.length; i++ ) {
                assert n.parents[ i ].payload != null
                boolean[] exists = {false}
                n.parents[ i ].forEachTarget({ t -> exists[0] = true })
                assert exists[0]
            }

        when:
            a = null
            b = null
            c = null
            d = null
            e = null
            System.gc()
            Thread.sleep(100)
            System.gc()
            Thread.sleep(200)

        then :
            for ( GraphNode p : n.parents ) {
                assert p.payload == null
                assert !p.hasDerivatives()
            }
    }


}
