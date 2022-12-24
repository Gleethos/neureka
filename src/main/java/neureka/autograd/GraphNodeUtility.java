package neureka.autograd;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.devices.Device;

/**
 *  This class exists in order to allow for {@link GraphNode}s to be instantiated
 *  with final field variables by collecting them when defined
 *  within constructor methods...
 */
final class GraphNodeUtility {

    private GraphNodeUtility() {}

    /**
     *  Evaluates and sets the auto-grad/auto-differentiation mode:
     *  A positive value means that the AD-procedure will be forward mode AD,
     *  whereas a negative value is backward mode AD.
     *  If the resulting mode equals 0 then this means that no auto differentiation is needed.
     *  This class tries to optimize the calculation of partial derivatives by forward propagating them
     *  for as long as only a single input for every computation graph node requires gradients,
     *  and they all are differentiable!
     *
     */
    public static <V> int modeOf( AutoDiffMode adMode, ExecutionCall<? extends Device<?>> call )
    {
        Tsr<V>[] inputs = (Tsr<V>[]) call.inputs();
        int resultMode = 0;
        int[] modes = new int[ inputs.length ];
        int inputMode = 0;
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode<V> node = inputs[ i ].getGraphNode().orElseThrow(IllegalStateException::new); // Not null checked in constructor!
            modes[ i ] = ( inputs[ i ].rqsGradient() ) ? 1 : node.getMode();
            inputMode += ( modes[ i ] != 0) ? 1 : 0;
        }
        if ( inputMode == 1 && adMode.allowsForward() ) { // Convolution and reshaping prohibit forward AutoDiff
            for ( int i = 0; i < inputs.length; i++ ) {
                resultMode +=
                        ( modes[ i ] == 0 )
                                ? 0
                                : ( modes[ i ] < 0 ) ? 1 : modes[ i ] + 1;
            }
        } // Reverse mode auto-differentiation :
        else if ( adMode.allowsBackward() ) resultMode = -inputMode;

        return resultMode;
    }

}
