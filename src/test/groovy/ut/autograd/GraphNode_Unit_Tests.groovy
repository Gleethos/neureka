package ut.autograd

import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.host.HostCPU
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
import spock.lang.Specification

import java.util.function.Supplier

class GraphNode_Unit_Tests extends Specification
{

    def 'GraphNode instantiation works as expected.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock(Tsr)
            Supplier<Tsr> supplier = () -> payload
            Function function = Mock(Function)
            Object context = Mock(GraphLock)
            GraphNode node
            function.doesAD() >> true
            payload.device() >> HostCPU.instance()

        when : 'We try to instantiate a GraphNode...'
            new GraphNode(function, context, supplier)

        then : 'The Tsr mock has been called.'
            1 * payload.add( _ )
    }



}
