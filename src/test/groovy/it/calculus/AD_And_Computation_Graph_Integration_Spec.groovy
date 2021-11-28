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
            it.scientific( true )
            it.multiline( false )
            it.withGradient( true )
            it.withCellSize( 1 )
            it.withValue( true )
            it.withRecursiveGraph( false )
            it.withDerivatives( true )
            it.withShape( true )
            it.cellBound( false )
            it.withPostfix(  "" )
            it.withPrefix(  ""  )
            it.withSlimNumbers(  false )  
        })
    }

    def "Reshaping produces expected computation graph and also works with reverse mode AD."(){

        given :
            Neureka.get().settings().view().getTensorSettings().legacy(true)
            Tsr a = Tsr.of([2, 3], [
                    1, 2, 3,
                    4, 5, 6
            ]).setRqsGradient(true)
            Function rs = Function.of("[1, 0]:(I[0])")

        when :
            Tsr b = rs(a)
            GraphNode na = a.get( GraphNode.class )
            GraphNode nb = b.get( GraphNode.class )

        then :
            assert na.getChildren().size()==1
            assert b.toString().contains("")//Todo!
            b.backward(Tsr.of([3, 2],[
                    -1, 2,
                    4, 7,
                    -9, 8
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
            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = a * 3 / 5
            Tsr c = Tsr.of(3)
            Tsr d = b ^ c
            Tsr e = d * c
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
