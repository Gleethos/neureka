package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.NDConvolution;
import neureka.backend.main.implementations.convolution.CLConvolution;
import neureka.backend.main.implementations.convolution.CPUConvolution;
import neureka.backend.main.operations.ConvUtil;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class Convolution extends AbstractOperation
{
    public Convolution()
    {
        super(
                new OperationBuilder()
                        .identifier(       "mul_conv"  )
                        .operator(         "x"         )
                        .arity(            2           )
                        .isOperator(       true        )
                        .isIndexer(        false       )
                        .isDifferentiable( true        )
                        .isInline(         false       )
        );

        setAlgorithm(
            NDConvolution.class,
            new NDConvolution()
                .setAutogradModeFor( call -> {
                    if ( call.getOperation().supports( NDConvolution.class ) ) return AutoDiffMode.BACKWARD_ONLY;
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiffMode.BACKWARD_ONLY;
                        last = t; // Note: shapes are cached!
                    }
                    return AutoDiffMode.FORWARD_AND_BACKWARD;
                })
                .setDeviceExecution(
                    (call, executor) ->
                    {
                        Tsr<?>[] tensors = call.inputs();
                        for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );

                        ExecutionCall<?> prepared = AbstractDeviceAlgorithm._prepareForExecution( call.withInputs(tensors) );
                        return AbstractDeviceAlgorithm.executeOnCommonDevice(
                                prepared,()->ConvUtil.executeRecursively( "x", prepared, null/*recursion is not expected to happen here*/ )
                        );
                    },
                    ( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                    {
                        int d = adCall.getDerivativeIndex();
                        Function deConv = new FunctionParser( Neureka.get().backend() ).parse(
                                "I[ 0 ] x>> I[ 1 ] x>> I[ 2 ]",
                                false
                        );
                        Tsr<?> derivative = f.derive( (Tsr[]) adCall.inputs(), d );
                        assert d >= 0 && d <= 1;
                        assert derivative != null;
                        assert deConv != null;
                        assert adCall.arity() >= 2 && adCall.arity() <= 3;
                        // Now we need to remember the shape of the input which is targeted for back prop.
                        int[] shape = adCall.input( adCall.arity() > 2 ? d + 1 : d ).getNDConf().shape();
                        // This is because it will be the shape of the output to the de-convolution!
                        return ADAction.of( target ->
                                                deConv.execute(
                                                    target.error(),
                                                    derivative,
                                                    Tsr.of(shape, 0).getUnsafe().setIsIntermediate( false )
                                                )
                                        );
                    }
                )
                .setCallPreparation(
                     call -> {
                         Device<Number> device = call.getDeviceFor(Number.class);
                         int[] shp = ConvUtil.shapeOfCon(call.input( 1 ).getNDConf().shape(), call.input( 2 ).getNDConf().shape());
                         Tsr<Number> output = (Tsr<Number>) Tsr.of( call.input(1).getItemType(), shp, 0 )
                                                 .getUnsafe()
                                                 .setIsIntermediate( true );
                         output.setIsVirtual( false );
                         //device.store( output );//Todo: find out why this causes problems
                         return call.withInputAt( 0, output );
                     }
                )
                .buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    new CPUConvolution()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    new CLConvolution( this.getIdentifier() )
                )
        );

    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
