package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.ConvUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class XConv extends AbstractOperation
{
    public XConv()
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
            Convolution.class,
            new Convolution()
                .setAutogradModeFor( call -> {
                    if ( call.getOperation().supports( Convolution.class ) ) return AutoDiffMode.BACKWARD_ONLY;
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiffMode.BACKWARD_ONLY;
                        last = t; // Note: shapes are cached!
                    }
                    return AutoDiffMode.FORWARD_AND_BACKWARD;
                })
                .setDeviceExecution(
                    (context, executor) ->
                    {
                        ExecutionCall<?> call = context.initialCall();
                        Tsr<?>[] tensors = new Tsr[]{null, call.input( 0 ), call.input( 1 )};
                        tensors[ 0 ] =
                            (call.getValOf( Arg.DerivIdx.class ) < 0)
                                    ? Tsr.of(
                                            call.input(0).getItemType(),
                                            ConvUtil.shapeOfCon( tensors[ 1 ].getNDConf().shape(), tensors[ 2 ].getNDConf().shape() ),
                                            0
                                    )
                                    .getUnsafe()
                                    .setIsIntermediate( true )
                                    : null;

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
                        return ADAgent.of( derivative )
                                .withAD( target ->
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
                         if ( call.input( 0 ) == null ) // Creating a new tensor:
                         {
                             int[] shp = ConvUtil.shapeOfCon(call.input( 1 ).getNDConf().shape(), call.input( 2 ).getNDConf().shape());
                             Tsr<Double> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                             output.setIsVirtual( false );
                             device.store( output );
                             return call.withInputAt( 0, output );
                         }
                         return call;
                     }
                )
                .buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    Convolution.implementationForCPU()
                            .with(Fun.F64F64ToF64.triple(
                                    ( a, b ) -> a * b,
                                    ( a, b ) -> b, // Deriving at input 0
                                    ( a, b ) -> a  // deriving input 1
                            ))
                            .with(Fun.F32F32ToF32.triple(
                                    ( a, b ) -> a * b,
                                    ( a, b ) -> b, // Deriving at input 0
                                    ( a, b ) -> a  // deriving input 1
                            ))
                            .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    CLImplementation.compiler()
                            .arity( 3 )
                            .kernelSource( Neureka.get().utility().readResource("kernels/convolution_template.cl") )
                            .activationSource( "value = src1 * src2;\n" )
                            .differentiationSource( "value += handle * drain;\n" )
                            .kernelPostfix( this.getIdentifier() )
                            .execution(
                                call -> {
                                    int offset = ( call.input( Number.class, 0 ) != null ) ? 0 : 1;
                                    int gwz = ( call.input( Number.class, 0 ) != null ) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                    call.getDevice()
                                        .getKernel(call)
                                        .passAllOf( call.input( Number.class, offset ) )
                                        .passAllOf( call.input( Number.class, offset + 1 ) )
                                        .passAllOf( call.input( Number.class, offset + 2 ) )
                                        .pass( call.input( Number.class, 0 ).rank() )
                                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                                        .call( gwz );

                                    return call.input( 0 );
                                }
                            )
                            .build()
                )
        );

    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
