package ut.autograd

import neureka.Neureka
import neureka.Tensor
import neureka.autograd.GraphNode
import neureka.math.Function
import neureka.view.NDPrintSettings
import spock.lang.Specification
import testutility.Sleep


class AD_And_Computation_Graph_Spec extends Specification
{
    def setup()
    {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
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

    def "Reshaping produces expected computation graph and also works with reverse mode AD."()
    {
        given :
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            Tensor<Double> a = Tensor.of([2, 3], [
                                1d, 2d, 3d,
                                4d, 5d, 6d
                            ]).setRqsGradient(true)
        and :
            Function rs = Function.of("[1, 0]:(I[0])")

        when :
            Tensor b = rs(a)
            GraphNode na = a.get( GraphNode.class )
            GraphNode nb = b.get( GraphNode.class )

        then :
            na.getChildren().size() == 1
            b.toString().contains("")//Todo!
            b.backward(Tensor.of([3, 2],[
                                -1d, 2d,
                                 4d, 7d,
                                -9d, 8d
                        ]))
            a.toString().contains("[2x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0):g:(-1.0, 4.0, -9.0, 2.0, 7.0, 8.0)")
            b.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")
            na.isLeave()
            !na.function.isPresent()
            na.parents == []
            !na.pendingError.isPresent()
            na.getMode() == 1
        and : 'We expect the partial derivative to be cleaned up! (size == 0)'
            na.size()==0
            !nb.isLeave()
            nb.function.isPresent()
            nb.getMode() == -1
            nb.size()==0
    }

    def "Payloads and derivatives are null after garbage collection."()
    {
        given :
            Tensor<Double> a = Tensor.of(2d).setRqsGradient(true)
            Tensor<Double> b = a * 3 / 5
            Tensor<Double> c = Tensor.of(3d)
            Tensor<Double> d = b ** c
        Tensor<Double> e = d * c
            GraphNode n = e.get( GraphNode.class )
            var strongRefs = n.parents.collect { it.payload.get() }

        expect :
            !n.parents[0].isLeave()
            n.parents[0].isGraphLeave()
            n.parents[1].isLeave()
            n.parents[1].isGraphLeave()

        and :
            for ( int i = 0; i < n.parents.size(); i++ ) {
                assert n.parents[ i ].payload.isPresent()
                boolean[] exists = {false}
                n.parents[ i ].forEachTarget({ t -> exists[0] = true })
                assert exists[0]
            }

        when:
            strongRefs = null
            a = null
            b = null
            c = null
            d = null
            e = null
            System.gc()
            Sleep.until(220, {
                n.parents.every {!it.payload.isPresent() && !it.hasDerivatives()}
            })
            System.gc()
            Sleep.until(220, {
                n.parents.every {!it.payload.isPresent() && !it.hasDerivatives()}
            })
            System.gc()

        then :
            for ( GraphNode p : n.parents )
                assert !p.payload.isPresent()
    }


}
